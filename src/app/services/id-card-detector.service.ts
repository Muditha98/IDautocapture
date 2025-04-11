// src/app/services/id-card-detector.service.ts
// Use the global ort variable
declare const ort: any;

import { Injectable } from '@angular/core';

export interface Detection {
  box: number[];
  confidence: number;
  class_id: number;
  class_name: string;
}

@Injectable({
  providedIn: 'root'
})
export class IdCardDetectorService {
  private session: any = null;
  private inputName: string = '';
  private outputName: string = '';
  private readonly className: string = 'Identity-cards';
  private inputSize: [number, number] = [640, 640];
  private confThreshold: number = 0.25;
  private iouThreshold: number = 0.45;
  private origWidth: number = 0;
  private origHeight: number = 0;

  constructor() {}

  /**
   * Initialize the ONNX model
   * @param modelData - Can be a path string or ArrayBuffer containing the model data
   */
  async initialize(modelData: string | ArrayBuffer): Promise<void> {
    try {
      console.log('Starting ONNX model initialization...');
      
      // Create the ONNX session with minimal options
      const options = {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'basic',
        enableCpuMemArena: false
      };
      
      // Create the session
      this.session = await ort.InferenceSession.create(modelData, options);
      
      // Get input and output names
      this.inputName = this.session.inputNames[0];
      this.outputName = this.session.outputNames[0];
      
      console.log('ONNX model initialized successfully');
      console.log('Input name:', this.inputName);
      console.log('Output name:', this.outputName);
    } catch (error) {
      console.error('Error initializing ONNX model:', error);
      throw error;
    }
  }

  /**
   * Preprocess the input image for the ONNX model
   */
  private preprocess(imageData: ImageData): any {
    // Store original dimensions
    this.origWidth = imageData.width;
    this.origHeight = imageData.height;
    
    // Create a canvas to resize the image
    const canvas = document.createElement('canvas');
    canvas.width = this.inputSize[0];
    canvas.height = this.inputSize[1];
    const ctx = canvas.getContext('2d');
    
    if (!ctx) {
      throw new Error('Could not get canvas context');
    }
    
    // Create an image from the ImageData
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = imageData.width;
    tempCanvas.height = imageData.height;
    const tempCtx = tempCanvas.getContext('2d');
    
    if (!tempCtx) {
      throw new Error('Could not get temporary canvas context');
    }
    
    tempCtx.putImageData(imageData, 0, 0);
    
    // Resize the image to the input size
    ctx.drawImage(tempCanvas, 0, 0, imageData.width, imageData.height, 0, 0, this.inputSize[0], this.inputSize[1]);
    
    // Get the resized image data
    const resizedImageData = ctx.getImageData(0, 0, this.inputSize[0], this.inputSize[1]);
    
    // Normalize pixel values to 0-1 and convert to NCHW format (batch, channels, height, width)
    const inputTensor = new Float32Array(1 * 3 * this.inputSize[0] * this.inputSize[1]);
    
    // RGB order for YOLOv8 (OpenCV uses BGR)
    for (let y = 0; y < this.inputSize[1]; y++) {
      for (let x = 0; x < this.inputSize[0]; x++) {
        const pixelIndex = (y * this.inputSize[0] + x) * 4;
        const r = resizedImageData.data[pixelIndex] / 255.0;
        const g = resizedImageData.data[pixelIndex + 1] / 255.0;
        const b = resizedImageData.data[pixelIndex + 2] / 255.0;
        
        // NCHW format - batch 0, channel R, Y, X
        inputTensor[0 * this.inputSize[0] * this.inputSize[1] + y * this.inputSize[0] + x] = r;
        // NCHW format - batch 0, channel G, Y, X
        inputTensor[1 * this.inputSize[0] * this.inputSize[1] + y * this.inputSize[0] + x] = g;
        // NCHW format - batch 0, channel B, Y, X
        inputTensor[2 * this.inputSize[0] * this.inputSize[1] + y * this.inputSize[0] + x] = b;
      }
    }
    
    // Create ONNX tensor with shape [1, 3, height, width]
    return new ort.Tensor('float32', inputTensor, [1, 3, this.inputSize[0], this.inputSize[1]]);
  }

  /**
   * Process the model output to get bounding boxes, confidence scores, and class IDs
   */
  private postprocess(outputs: any): Detection[] {
    if (!outputs) {
      return [];
    }
    
    // Get the output data
    const output = outputs.data;
    
    // YOLOv8 output format: [batch, num_classes+4, num_anchors]
    // Reshape the output to [batch, num_anchors, num_classes+4]
    const batch = 1;
    const numAnchors = 8400; // YOLOv8 default
    const dimensions = 5; // 4 box coordinates + 1 class score
    
    // Create new array to store reshaped data
    const reshapedOutput = new Float32Array(batch * numAnchors * dimensions);
    
    // Transpose the data from [batch, dimensions, anchors] to [batch, anchors, dimensions]
    for (let b = 0; b < batch; b++) {
      for (let a = 0; a < numAnchors; a++) {
        for (let d = 0; d < dimensions; d++) {
          // Original: [b, d, a]
          const srcIdx = b * dimensions * numAnchors + d * numAnchors + a;
          // Target: [b, a, d]
          const dstIdx = b * numAnchors * dimensions + a * dimensions + d;
          reshapedOutput[dstIdx] = output[srcIdx];
        }
      }
    }
    
    // Process detections for first batch
    const batchOutput = Array.from({ length: numAnchors }, (_, i) => {
      const offset = i * dimensions;
      return {
        x: reshapedOutput[offset],
        y: reshapedOutput[offset + 1],
        w: reshapedOutput[offset + 2],
        h: reshapedOutput[offset + 3],
        confidence: reshapedOutput[offset + 4]
      };
    });
    
    // Filter by confidence threshold
    const filteredBoxes = batchOutput.filter(box => box.confidence >= this.confThreshold);
    
    if (filteredBoxes.length === 0) {
      return [];
    }
    
    // Convert YOLOv8 format [x, y, w, h] (center, width, height) to [x1, y1, x2, y2] (corners)
    const boxes = filteredBoxes.map(box => {
      // Convert center coordinates to top-left and bottom-right corners
      const x1 = (box.x - box.w / 2) * this.origWidth / this.inputSize[0];
      const y1 = (box.y - box.h / 2) * this.origHeight / this.inputSize[1];
      const x2 = (box.x + box.w / 2) * this.origWidth / this.inputSize[0];
      const y2 = (box.y + box.h / 2) * this.origHeight / this.inputSize[1];
      
      return {
        box: [x1, y1, x2, y2],
        confidence: box.confidence
      };
    });
    
    // Apply NMS (Non-Maximum Suppression)
    const indices = this.nonMaxSuppression(
      boxes.map(b => b.box),
      boxes.map(b => b.confidence)
    );
    
    // Create detection results
    return indices.map(idx => ({
      box: boxes[idx].box.map(Math.round),
      confidence: boxes[idx].confidence,
      class_id: 0,
      class_name: this.className
    }));
  }

  /**
   * Perform Non-Maximum Suppression to filter out overlapping bounding boxes
   */
  private nonMaxSuppression(boxes: number[][], scores: number[]): number[] {
    // Get the coordinates of bounding boxes
    const x1 = boxes.map(box => box[0]);
    const y1 = boxes.map(box => box[1]);
    const x2 = boxes.map(box => box[2]);
    const y2 = boxes.map(box => box[3]);
    
    // Calculate area of each box
    const areas = boxes.map(box => (box[2] - box[0]) * (box[3] - box[1]));
    
    // Sort by confidence score in descending order
    const order = Array.from(scores.keys())
      .sort((a, b) => scores[b] - scores[a]);
    
    const keep: number[] = [];
    
    while (order.length > 0) {
      // Pick the box with highest confidence score
      const i = order[0];
      keep.push(i);
      
      // Calculate IoU of the picked box with rest of the boxes
      const remainingIndices = order.slice(1);
      
      // Calculate intersection coordinates
      const xx1 = remainingIndices.map(j => Math.max(x1[i], x1[j]));
      const yy1 = remainingIndices.map(j => Math.max(y1[i], y1[j]));
      const xx2 = remainingIndices.map(j => Math.min(x2[i], x2[j]));
      const yy2 = remainingIndices.map(j => Math.min(y2[i], y2[j]));
      
      // Calculate width and height of intersection
      const w = remainingIndices.map((_, idx) => Math.max(0, xx2[idx] - xx1[idx]));
      const h = remainingIndices.map((_, idx) => Math.max(0, yy2[idx] - yy1[idx]));
      
      // Calculate intersection area
      const inter = remainingIndices.map((_, idx) => w[idx] * h[idx]);
      
      // Calculate IoU (intersection over union)
      const iou = remainingIndices.map((j, idx) => {
        return inter[idx] / (areas[i] + areas[j] - inter[idx]);
      });
      
      // Keep boxes with IoU less than threshold
      const filteredIndices = remainingIndices.filter((_, idx) => iou[idx] <= this.iouThreshold);
      
      // Update order array
      order.length = 0;
      order.push(...filteredIndices);
    }
    
    return keep;
  }

  /**
   * Run detection on an image
   */
  async detect(imageData: ImageData): Promise<{ detections: Detection[]; inferenceTime: number }> {
    if (!this.session) {
      throw new Error('ONNX model not initialized');
    }
    
    // Preprocess image
    const inputTensor = this.preprocess(imageData);
    
    // Run inference
    const startTime = performance.now();
    const outputs = await this.session.run({ [this.inputName]: inputTensor });
    const inferenceTime = performance.now() - startTime;
    
    // Postprocess outputs
    const detections = this.postprocess(outputs[this.outputName]);
    
    return {
      detections,
      inferenceTime
    };
  }
}