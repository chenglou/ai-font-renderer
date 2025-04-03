import { createCanvas } from "canvas"
import fs from "fs"
import path from "path"

// Function to create a BMP file buffer directly
function createBMPBuffer(data: Uint8ClampedArray, width: number, height: number): Buffer {
  // BMP file header (14 bytes)
  const fileHeaderSize = 14
  // DIB header (40 bytes for BITMAPINFOHEADER)
  const dibHeaderSize = 40
  // Each pixel is 3 bytes (BGR)
  const pixelSize = 3
  // Pad rows to multiples of 4 bytes
  const rowSize = Math.floor((width * pixelSize + 3) / 4) * 4
  const dataSize = rowSize * height
  // Total file size
  const fileSize = fileHeaderSize + dibHeaderSize + dataSize

  // Create buffer for the entire file
  const buffer = Buffer.alloc(fileSize)

  // File header (14 bytes)
  buffer.write('BM', 0) // Signature
  buffer.writeUInt32LE(fileSize, 2) // File size
  buffer.writeUInt32LE(0, 6) // Reserved
  buffer.writeUInt32LE(fileHeaderSize + dibHeaderSize, 10) // Offset to pixel data

  // DIB header (40 bytes) - BITMAPINFOHEADER
  buffer.writeUInt32LE(dibHeaderSize, 14) // Header size
  buffer.writeInt32LE(width, 18) // Width
  buffer.writeInt32LE(-height, 22) // Height (negative for top-down)
  buffer.writeUInt16LE(1, 26) // Planes
  buffer.writeUInt16LE(24, 28) // Bits per pixel (24 for BGR)
  buffer.writeUInt32LE(0, 30) // Compression (0 = none)
  buffer.writeUInt32LE(dataSize, 34) // Image size
  buffer.writeInt32LE(0, 38) // X pixels per meter
  buffer.writeInt32LE(0, 42) // Y pixels per meter
  buffer.writeUInt32LE(0, 46) // Total colors
  buffer.writeUInt32LE(0, 50) // Important colors

  // Write pixel data (BGR format, top-down)
  const pixelOffset = fileHeaderSize + dibHeaderSize

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4 // RGBA data
      const offset = pixelOffset + y * rowSize + x * pixelSize

      // Write BGR (reverse of RGB)
      buffer[offset] = data[idx + 2] // B
      buffer[offset + 1] = data[idx + 1] // G
      buffer[offset + 2] = data[idx] // R
    }

    // Add padding to make row size a multiple of 4
    for (let p = width * pixelSize; p < rowSize; p++) {
      buffer[pixelOffset + y * rowSize + p] = 0
    }
  }

  return buffer
}

// Font configuration
const FONT_PATH = "FiraCode-Retina.ttf"
const FONT_SIZE = 24
const FONT_FAMILY = "Fira Code"

// Sheet dimensions
const SHEET_WIDTH = 480
const SHEET_HEIGHT = 160
const PADDING = 0

// Function to wrap text with word breaks
function wrapText(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string[] {
  const words = text.split(" ")
  const lines = []
  let currentLine = ""

  for (const word of words) {
    const testLine = currentLine ? `${currentLine} ${word}` : word
    const metrics = ctx.measureText(testLine)

    if (metrics.width > maxWidth && currentLine) {
      lines.push(currentLine)
      currentLine = word
    } else {
      currentLine = testLine
    }
  }

  if (currentLine) {
    lines.push(currentLine)
  }

  return lines
}

// Create a reusable canvas
const canvas = createCanvas(SHEET_WIDTH, SHEET_HEIGHT)
const ctx = canvas.getContext("2d")

// Register font once (important for non-standard fonts)
try {
  const { registerFont } = require('canvas')
  registerFont(FONT_PATH, { family: FONT_FAMILY })
} catch (err) {
  console.warn(`Could not register font: ${err.message}`)
}

// Render text to bitmap using the font and save to file
function renderTextToBitmap(text: string, outputPath: string): void {
  // Clear canvas with white background
  ctx.fillStyle = "white"
  ctx.fillRect(0, 0, SHEET_WIDTH, SHEET_HEIGHT)

  // Configure font - needs to be set each time as ctx state could change
  ctx.font = `${FONT_SIZE}px "${FONT_FAMILY}"`
  ctx.fillStyle = "black"

  // Word-wrap the text
  const lines = wrapText(ctx, text, SHEET_WIDTH - PADDING * 2)

  // Calculate line height (add some extra spacing)
  const lineHeight = FONT_SIZE * 1.2

  // Draw each line
  lines.forEach((line, index) => {
    ctx.fillText(line, PADDING, PADDING + (index + 1) * lineHeight)
  })

  // Ensure directory exists
  const dir = path.dirname(outputPath)
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true })
  }

  // Save as BMP
  const imageData = ctx.getImageData(0, 0, SHEET_WIDTH, SHEET_HEIGHT)
  const bmpBuffer = createBMPBuffer(imageData.data, SHEET_WIDTH, SHEET_HEIGHT)
  fs.writeFileSync(outputPath, bmpBuffer)
}

// Create dataset directory
const SAMPLES_DIR = "train_input"

// Remove existing directory to avoid stale data
if (fs.existsSync(SAMPLES_DIR)) {
  console.log(`Removing existing directory ${SAMPLES_DIR}...`)
  fs.rmSync(SAMPLES_DIR, { recursive: true, force: true })
}

// Create fresh directory
fs.mkdirSync(SAMPLES_DIR, { recursive: true })

// Generate samples
console.log(`Generating text samples with Fira Code font in ${SAMPLES_DIR}/...`)

const numSamples = 50000
const textData = []
const renderQueue = []

// Create pseudorandom number generator with seed
function createSeededRandom(seed: number) {
  return function() {
    // Simple LCG (Linear Congruential Generator)
    seed = (seed * 1664525 + 1013904223) % 4294967296
    return seed / 4294967296 // Normalize to [0, 1)
  }
}

// Generate seeded random text based on an index
function generateSeededRandomText(seed: number, minLength: number, maxLength: number): string {
  const random = createSeededRandom(seed)

  const words = []
  const length = Math.floor(random() * (maxLength - minLength + 1)) + minLength

  let remainingChars = length
  while (remainingChars > 0) {
    // Generate a random word with length between 1 and 15 characters
    const randomWordLength = Math.min(Math.floor(random() * 10) + 1, remainingChars)
    let word = ""
    for (let i = 0; i < randomWordLength; i++) {
      const randomChar = String.fromCharCode(65 + Math.floor(random() * 26)) // A-Z
      word += randomChar
    }
    words.push(word)
    remainingChars -= randomWordLength

    // Add space if there's room
    if (remainingChars > 0) {
      words.push(" ")
      remainingChars--
    }
  }

  return words.join("")
}

// Generate all texts with seeded randomness (fast sync operation)
console.log("Generating seeded random texts...")
for (let i = 0; i < numSamples; i++) {
  const seed = i + 42 // Base seed + index, ensuring reproducibility
  const text = generateSeededRandomText(seed, 20, 130)
  textData.push(text)

  // Queue up the rendering operation (will be done asynchronously)
  renderQueue.push({
    index: i + 1, // Use 1-based indexing for files
    text
  })
}

// Save all text data to a single file
fs.writeFileSync(`${SAMPLES_DIR}/data.txt`, textData.join("\n"))
console.log(`Saved all text data to ${SAMPLES_DIR}/data.txt`)

// Process the render queue
console.log("Rendering bitmaps...")
for (const item of renderQueue) {
  const outputPath = `${SAMPLES_DIR}/${item.index}.bmp`
  renderTextToBitmap(item.text, outputPath)
}

// Create metadata file
const metadata = `AI Font Renderer Dataset - Fira Code
==============================

Font: ${FONT_PATH}
Font size: ${FONT_SIZE}
Sheet dimensions: ${SHEET_WIDTH}x${SHEET_HEIGHT}
Padding: ${PADDING}px

Format: Images are numbered sequentially (1.bmp, 2.bmp, etc.)
Text labels are stored line by line in data.txt (line 1 corresponds to 1.bmp)
`

fs.writeFileSync(`${SAMPLES_DIR}/dataset_metadata.txt`, metadata)

console.log(`Dataset generation complete. Check the ${SAMPLES_DIR}/ directory.`)
