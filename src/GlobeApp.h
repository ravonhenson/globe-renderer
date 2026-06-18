#pragma once

#include "HexGrid.h"
#include "QuadtreeMesh.h"
#include "VulkanUtils.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <future>
#include <optional>
#include <vector>

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() const {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

// Result of a (possibly background) quadtree mesh rebuild: the selected
// leaves plus their tessellated vertex/index data, ready to be memcpy'd
// into the GPU-mapped buffers on the render thread.
struct MeshBuildResult {
    std::vector<QuadtreeMesh::Patch> leaves;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
};

// Owns the window, the Vulkan context, the scene's GPU resources, and the
// per-frame render loop for the globe renderer.
class GlobeApp {
public:
    explicit GlobeApp(std::filesystem::path executablePath);

    void run();

private:
    GLFWwindow* window_ = nullptr;

    VkInstance instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties physicalDeviceProperties_{};
    VkPhysicalDeviceFeatures enabledFeatures_{};
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue graphicsQueue_ = VK_NULL_HANDLE;
    VkQueue presentQueue_ = VK_NULL_HANDLE;

    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    std::vector<VkImage> swapchainImages_;
    VkFormat swapchainImageFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchainExtent_{};
    std::vector<VkImageView> swapchainImageViews_;

    VkRenderPass renderPass_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline_ = VK_NULL_HANDLE;
    VkPipeline wireframePipeline_ = VK_NULL_HANDLE;

    VkFormat depthFormat_ = VK_FORMAT_UNDEFINED;
    VkImage depthImage_ = VK_NULL_HANDLE;
    VkDeviceMemory depthImageMemory_ = VK_NULL_HANDLE;
    VkImageView depthImageView_ = VK_NULL_HANDLE;

    std::vector<VkFramebuffer> swapchainFramebuffers_;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers_;

    // Number of frames that may be in flight at once (matches the
    // swapchain's sync object count). Vertex/index buffers are
    // double-buffered, one set per in-flight frame slot, so updateMesh() can
    // safely overwrite a slot's buffers via memcpy as soon as
    // vkWaitForFences confirms the GPU is done with that slot's prior frame.
    static constexpr size_t kMaxFramesInFlight = 2;

    // Persistently-mapped, host-visible vertex/index buffers sized for
    // QuadtreeMesh::kMaxVertices/kMaxIndices, one set per in-flight frame
    // slot. Rewritten in place by updateMesh() whenever the active leaf set
    // changes.
    std::array<VkBuffer, kMaxFramesInFlight> vertexBuffers_{};
    std::array<VkDeviceMemory, kMaxFramesInFlight> vertexBufferMemories_{};
    std::array<void*, kMaxFramesInFlight> vertexBuffersMapped_{};
    std::array<VkBuffer, kMaxFramesInFlight> indexBuffers_{};
    std::array<VkDeviceMemory, kMaxFramesInFlight> indexBufferMemories_{};
    std::array<void*, kMaxFramesInFlight> indexBuffersMapped_{};
    std::array<uint32_t, kMaxFramesInFlight> indexCounts_{};
    std::vector<QuadtreeMesh::Patch> activeLeafPatches_;
    // Object-space camera position at the last LOD rebuild kickoff, used to
    // skip starting a new rebuild for tiny camera movements. Initialized
    // far from any real camera position so the first call to updateMesh()
    // always rebuilds.
    glm::vec3 lastLodCameraPos_{1e9f, 1e9f, 1e9f};
    // Quadtree mesh rebuilds (LOD selection + retessellation) are expensive
    // enough that doing them synchronously on the render thread causes
    // visible frame drops during continuous camera movement. updateMesh()
    // instead runs at most one rebuild at a time on a background thread and
    // swaps the result into the GPU-mapped buffers once it's ready.
    std::future<MeshBuildResult> meshFuture_;
    // A finished background build, staged for propagation into every
    // in-flight frame slot's buffers (one slot per updateMesh() call) so the
    // GPU never observes a partially-overwritten buffer.
    std::optional<MeshBuildResult> pendingMeshResult_;
    int pendingMeshWritesRemaining_ = 0;

    uint32_t textureMipLevels_ = 1;
    VkImage textureImage_ = VK_NULL_HANDLE;
    VkDeviceMemory textureImageMemory_ = VK_NULL_HANDLE;
    VkImageView textureImageView_ = VK_NULL_HANDLE;
    VkSampler textureSampler_ = VK_NULL_HANDLE;

    VkImage heightImage_ = VK_NULL_HANDLE;
    VkDeviceMemory heightImageMemory_ = VK_NULL_HANDLE;
    VkImageView heightImageView_ = VK_NULL_HANDLE;
    VkSampler heightSampler_ = VK_NULL_HANDLE;

    std::vector<VkBuffer> uniformBuffers_;
    std::vector<VkDeviceMemory> uniformBuffersMemory_;
    std::vector<void*> uniformBuffersMapped_;

    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets_;

    std::vector<VkSemaphore> imageAvailableSemaphores_;
    std::vector<VkSemaphore> renderFinishedSemaphores_;
    std::vector<VkFence> inFlightFences_;
    std::vector<VkFence> imagesInFlight_;
    size_t currentFrame_ = 0;
    bool framebufferResized_ = false;
    std::filesystem::path executablePath_;

    // Camera/input tuning.
    static constexpr double kInitialCameraDistance = 2.5;
    static constexpr double kMinCameraDistance     = 1.001; // ~640 m above unit-sphere surface
    static constexpr double kMaxCameraDistance     = 6.0;
    // Altitude-proportional zoom: each scroll unit moves the camera by this
    // fraction of the current altitude above the surface. Keeps step size
    // physically sensible at every zoom level and is immune to large per-event
    // yoffset values from trackpads / high-resolution scroll wheels.
    static constexpr double kZoomSensitivity = 0.15;
    static constexpr float  kRotationSpeed = 0.005f;
    static constexpr float  kMaxPitch      = glm::radians(89.0f);

    // Hex overlay tuning.
    static constexpr double kHexOverlayThreshold = kMaxCameraDistance;
    static constexpr int    kHexSubdivisions     = 6;     // icosphere dual: N=6 → ~41k cells, circumradius ≈ 0.016
    static constexpr float  kHexNormalLength     = 0.05f; // outward extent in unit-sphere units

    // Minimum object-space camera movement (in unit-sphere units) before
    // the quadtree LOD is re-evaluated. Without this, the active leaf set
    // changes on almost every frame during rotation (the "near point" on
    // the sphere shifts continuously), forcing a full mesh retessellation
    // every frame.
    static constexpr float kLodRebuildThreshold = 0.05f;

    double cameraDistance_;
    float rotationYaw_ = 0.0f;
    float rotationPitch_ = 0.0f;
    bool dragging_ = false;
    double lastCursorX_ = 0.0;
    double lastCursorY_ = 0.0;
    bool wireframeEnabled_   = false;
    bool fpsCounterEnabled_  = false;
    bool hexOverlayEnabled_  = true;
    bool hexNormalsEnabled_  = false;

    // Hex overlay — full-sphere wireframe baked once at startup into a
    // device-local buffer. Rendered with the globe's model matrix so cells
    // are fixed features of the globe surface.
    VkBuffer       hexVertexBuffer_       = VK_NULL_HANDLE;
    VkDeviceMemory hexVertexBufferMemory_ = VK_NULL_HANDLE;
    uint32_t       hexVertexCount_        = 0;

    // Outward normals at each hex cell centre (toggle with N key).
    VkBuffer       hexNormalBuffer_       = VK_NULL_HANDLE;
    VkDeviceMemory hexNormalBufferMemory_ = VK_NULL_HANDLE;
    uint32_t       hexNormalCount_        = 0;

    // Hex overlay pipeline and layout (recreated with the swapchain).
    VkPipeline       hexPipeline_       = VK_NULL_HANDLE;
    VkPipelineLayout hexPipelineLayout_ = VK_NULL_HANDLE;

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);

    void initWindow();
    void initVulkan();
    void mainLoop();
    void cleanupSwapchain();
    void cleanup();
    void recreateSwapchain();

    void createInstance();
    void createSurface();
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) const;
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) const;
    bool isDeviceSuitable(VkPhysicalDevice device) const;
    void pickPhysicalDevice();
    void createLogicalDevice();

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) const;
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) const;
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) const;
    void createSwapchain();

    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels) const;
    void createImageViews();

    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) const;
    VkFormat findDepthFormat() const;
    void createRenderPass();
    void createDescriptorSetLayout();

    std::filesystem::path resolveResourcePath(const std::filesystem::path& relativePath) const;
    std::vector<char> loadShaderBinary(const std::filesystem::path& relativePath) const;
    VkShaderModule createShaderModule(const std::vector<char>& code) const;
    void createGraphicsPipeline();

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const;
    void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkFormat format, VkImageTiling tiling,
                      VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& memory) const;
    void createDepthResources();
    void createFramebuffers();

    void createCommandPool();
    VkCommandBuffer beginSingleTimeCommands() const;
    void endSingleTimeCommands(VkCommandBuffer commandBuffer) const;
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& memory) const;
    void copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size) const;
    void transitionImageLayout(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels) const;
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) const;
    void generateMipmaps(VkImage image, VkFormat format, int32_t texWidth, int32_t texHeight, uint32_t mipLevels) const;

    void createTextureImage();
    void createTextureImageView();
    void createTextureSampler();

    void createHeightmapImage();
    void createHeightmapImageView();
    void createHeightmapSampler();

    // Uploads `data` into a device-local buffer via a temporary host-visible
    // staging buffer. Templated on element type so it can build both the
    // vertex buffer (Vertex) and the index buffer (uint32_t); the generic
    // body means this stays defined here in the header.
    template <typename T>
    void createDeviceLocalBuffer(const std::vector<T>& data, VkBufferUsageFlags usage, VkBuffer& buffer, VkDeviceMemory& memory) const {
        const VkDeviceSize size = sizeof(T) * data.size();

        VkBuffer stagingBuffer = VK_NULL_HANDLE;
        VkDeviceMemory stagingBufferMemory = VK_NULL_HANDLE;
        createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      stagingBuffer, stagingBufferMemory);

        void* mapped = nullptr;
        checkVk(vkMapMemory(device_, stagingBufferMemory, 0, size, 0, &mapped), "Failed to map staging buffer");
        std::memcpy(mapped, data.data(), static_cast<size_t>(size));
        vkUnmapMemory(device_, stagingBufferMemory);

        createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer, memory);
        copyBuffer(stagingBuffer, buffer, size);

        vkDestroyBuffer(device_, stagingBuffer, nullptr);
        vkFreeMemory(device_, stagingBufferMemory, nullptr);
    }

    void createMeshBuffers();
    void initHexOverlay();
    void createHexPipeline();
    glm::mat4 computeModelRotation() const;
    glm::vec3 computeCameraObjectPos() const;
    void updateMesh();
    void createUniformBuffers();
    void updateUniformBuffer(uint32_t currentImage) const;

    void createDescriptorPool();
    void createDescriptorSets();

    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);
    void createCommandBuffers();
    void createSyncObjects();
    void drawFrame();
};
