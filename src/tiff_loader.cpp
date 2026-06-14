#include "tiff_loader.h"

#include <array>
#include <fstream>
#include <stdexcept>

namespace {

constexpr uint16_t kTagImageWidth = 256;
constexpr uint16_t kTagImageLength = 257;
constexpr uint16_t kTagBitsPerSample = 258;
constexpr uint16_t kTagCompression = 259;
constexpr uint16_t kTagPhotometric = 262;
constexpr uint16_t kTagStripOffsets = 273;
constexpr uint16_t kTagSamplesPerPixel = 277;
constexpr uint16_t kTagRowsPerStrip = 278;
constexpr uint16_t kTagStripByteCounts = 279;
constexpr uint16_t kTagPlanarConfig = 284;

constexpr uint16_t kTypeByte = 1;
constexpr uint16_t kTypeShort = 3;
constexpr uint16_t kTypeLong = 4;

struct RawIfdEntry {
    uint16_t tag = 0;
    uint16_t type = 0;
    uint32_t count = 0;
    std::array<uint8_t, 4> valueBytes{};
};

uint32_t typeSize(uint16_t type) {
    switch (type) {
        case kTypeByte:
            return 1;
        case kTypeShort:
            return 2;
        case kTypeLong:
            return 4;
        default:
            return 1;
    }
}

class TiffReader {
public:
    explicit TiffReader(const std::filesystem::path& path) : file_(path, std::ios::binary) {
        if (!file_) {
            throw std::runtime_error("Failed to open TIFF file: " + path.string());
        }
    }

    bool bigEndian() const { return bigEndian_; }

    uint16_t readU16() {
        uint8_t b[2];
        file_.read(reinterpret_cast<char*>(b), 2);
        return bigEndian_ ? static_cast<uint16_t>((b[0] << 8) | b[1])
                           : static_cast<uint16_t>((b[1] << 8) | b[0]);
    }

    uint32_t readU32() {
        uint8_t b[4];
        file_.read(reinterpret_cast<char*>(b), 4);
        if (bigEndian_) {
            return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | b[3];
        }
        return (uint32_t(b[3]) << 24) | (uint32_t(b[2]) << 16) | (uint32_t(b[1]) << 8) | b[0];
    }

    uint32_t decodeBytes(const uint8_t* bytes, uint32_t size) const {
        if (size == 1) {
            return bytes[0];
        }
        if (size == 2) {
            return bigEndian_ ? static_cast<uint32_t>((bytes[0] << 8) | bytes[1])
                               : static_cast<uint32_t>((bytes[1] << 8) | bytes[0]);
        }
        if (bigEndian_) {
            return (uint32_t(bytes[0]) << 24) | (uint32_t(bytes[1]) << 16) | (uint32_t(bytes[2]) << 8) | bytes[3];
        }
        return (uint32_t(bytes[3]) << 24) | (uint32_t(bytes[2]) << 16) | (uint32_t(bytes[1]) << 8) | bytes[0];
    }

    void seek(uint64_t offset) { file_.seekg(static_cast<std::streamoff>(offset)); }

    void read(char* dest, std::streamsize count) {
        file_.read(dest, count);
        if (!file_) {
            throw std::runtime_error("Unexpected end of TIFF file while reading pixel data");
        }
    }

    void detectByteOrder() {
        uint8_t marker[2];
        file_.read(reinterpret_cast<char*>(marker), 2);
        if (marker[0] == 'I' && marker[1] == 'I') {
            bigEndian_ = false;
        } else if (marker[0] == 'M' && marker[1] == 'M') {
            bigEndian_ = true;
        } else {
            throw std::runtime_error("Not a TIFF file: invalid byte order marker");
        }
    }

    // Reads the single value of an IFD entry whose count is 1.
    uint32_t scalarValue(const RawIfdEntry& entry) const {
        return decodeBytes(entry.valueBytes.data(), typeSize(entry.type));
    }

    // Reads an array of values for an IFD entry, following the offset if the
    // values do not fit inline within the 4-byte value field.
    std::vector<uint32_t> arrayValues(const RawIfdEntry& entry) {
        const uint32_t elemSize = typeSize(entry.type);
        const uint64_t totalSize = uint64_t(elemSize) * entry.count;

        std::vector<uint32_t> result(entry.count);
        if (totalSize <= 4) {
            for (uint32_t i = 0; i < entry.count; ++i) {
                result[i] = decodeBytes(entry.valueBytes.data() + i * elemSize, elemSize);
            }
            return result;
        }

        const uint32_t offset = decodeBytes(entry.valueBytes.data(), 4);
        const auto savedPos = file_.tellg();
        seek(offset);
        for (uint32_t i = 0; i < entry.count; ++i) {
            if (elemSize == 2) {
                result[i] = readU16();
            } else {
                result[i] = readU32();
            }
        }
        file_.seekg(savedPos);
        return result;
    }

private:
    std::ifstream file_;
    bool bigEndian_ = false;
};

} // namespace

RgbaImage loadTiffRgba(const std::filesystem::path& path, uint32_t maxDimension) {
    TiffReader reader(path);
    reader.detectByteOrder();

    const uint16_t magic = reader.readU16();
    if (magic != 42) {
        throw std::runtime_error("Not a TIFF file: invalid magic number");
    }

    const uint32_t ifdOffset = reader.readU32();
    reader.seek(ifdOffset);

    const uint16_t entryCount = reader.readU16();
    std::vector<RawIfdEntry> entries(entryCount);
    for (auto& entry : entries) {
        entry.tag = reader.readU16();
        entry.type = reader.readU16();
        entry.count = reader.readU32();
        // The value/offset field is always 4 bytes; read it byte-by-byte so it
        // can be reinterpreted either as an inline value or a file offset.
        char raw[4];
        reader.read(raw, 4);
        for (int i = 0; i < 4; ++i) {
            entry.valueBytes[i] = static_cast<uint8_t>(raw[i]);
        }
    }

    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t samplesPerPixel = 1;
    uint32_t compression = 1;
    uint32_t photometric = 0;
    uint32_t rowsPerStrip = 0;
    uint32_t planarConfig = 1;
    std::vector<uint32_t> bitsPerSample;
    std::vector<uint32_t> stripOffsets;
    std::vector<uint32_t> stripByteCounts;

    for (const auto& entry : entries) {
        switch (entry.tag) {
            case kTagImageWidth:
                width = reader.scalarValue(entry);
                break;
            case kTagImageLength:
                height = reader.scalarValue(entry);
                break;
            case kTagBitsPerSample:
                bitsPerSample = reader.arrayValues(entry);
                break;
            case kTagCompression:
                compression = reader.scalarValue(entry);
                break;
            case kTagPhotometric:
                photometric = reader.scalarValue(entry);
                break;
            case kTagStripOffsets:
                stripOffsets = reader.arrayValues(entry);
                break;
            case kTagSamplesPerPixel:
                samplesPerPixel = reader.scalarValue(entry);
                break;
            case kTagRowsPerStrip:
                rowsPerStrip = reader.scalarValue(entry);
                break;
            case kTagStripByteCounts:
                stripByteCounts = reader.arrayValues(entry);
                break;
            case kTagPlanarConfig:
                planarConfig = reader.scalarValue(entry);
                break;
            default:
                break;
        }
    }

    if (width == 0 || height == 0) {
        throw std::runtime_error("TIFF file is missing image dimensions");
    }
    if (compression != 1) {
        throw std::runtime_error("Unsupported TIFF: only uncompressed images are supported");
    }
    if (photometric != 2) {
        throw std::runtime_error("Unsupported TIFF: only RGB photometric interpretation is supported");
    }
    if (planarConfig != 1) {
        throw std::runtime_error("Unsupported TIFF: only chunky (interleaved) planar configuration is supported");
    }
    if (samplesPerPixel < 3) {
        throw std::runtime_error("Unsupported TIFF: expected at least 3 samples per pixel (RGB)");
    }
    for (uint32_t bits : bitsPerSample) {
        if (bits != 8) {
            throw std::runtime_error("Unsupported TIFF: only 8 bits per sample is supported");
        }
    }
    if (stripOffsets.empty() || stripByteCounts.empty()) {
        throw std::runtime_error("Unsupported TIFF: missing strip layout information");
    }
    if (rowsPerStrip == 0) {
        rowsPerStrip = height;
    }

    // Pick the smallest integer downsample factor that brings both
    // dimensions within the requested maximum.
    uint32_t factor = 1;
    while (width / factor > maxDimension || height / factor > maxDimension) {
        ++factor;
    }

    RgbaImage image;
    image.width = width / factor;
    image.height = height / factor;
    image.pixels.resize(uint64_t(image.width) * image.height * 4);

    const uint64_t srcRowStride = uint64_t(width) * samplesPerPixel;
    std::vector<uint8_t> rowBuffer(uint64_t(factor) * srcRowStride);

    for (uint32_t oy = 0; oy < image.height; ++oy) {
        for (uint32_t fy = 0; fy < factor; ++fy) {
            const uint32_t srcRow = oy * factor + fy;
            const uint32_t stripIndex = srcRow / rowsPerStrip;
            const uint32_t rowInStrip = srcRow % rowsPerStrip;
            const uint64_t fileOffset = uint64_t(stripOffsets[stripIndex]) + uint64_t(rowInStrip) * srcRowStride;

            reader.seek(fileOffset);
            reader.read(reinterpret_cast<char*>(rowBuffer.data() + uint64_t(fy) * srcRowStride), static_cast<std::streamsize>(srcRowStride));
        }

        for (uint32_t ox = 0; ox < image.width; ++ox) {
            uint32_t sum[3] = {0, 0, 0};
            for (uint32_t fy = 0; fy < factor; ++fy) {
                const uint8_t* row = rowBuffer.data() + uint64_t(fy) * srcRowStride;
                for (uint32_t fx = 0; fx < factor; ++fx) {
                    const uint8_t* px = row + uint64_t(ox * factor + fx) * samplesPerPixel;
                    sum[0] += px[0];
                    sum[1] += px[1];
                    sum[2] += px[2];
                }
            }

            const uint32_t sampleCount = factor * factor;
            uint8_t* dst = image.pixels.data() + (uint64_t(oy) * image.width + ox) * 4;
            dst[0] = static_cast<uint8_t>(sum[0] / sampleCount);
            dst[1] = static_cast<uint8_t>(sum[1] / sampleCount);
            dst[2] = static_cast<uint8_t>(sum[2] / sampleCount);
            dst[3] = 255;
        }
    }

    return image;
}
