// src/app/app.component.ts
import { Component, ElementRef, OnDestroy, OnInit, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { IdCardDetectorService, Detection } from './services/id-card-detector.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css'],
  standalone: true,
  imports: [CommonModule]
})
export class AppComponent implements OnInit, OnDestroy {
  @ViewChild('videoElement', { static: true }) videoElement!: ElementRef<HTMLVideoElement>;
  @ViewChild('canvasElement', { static: true }) canvasElement!: ElementRef<HTMLCanvasElement>;
  @ViewChild('shutterSound', { static: false }) shutterSound?: ElementRef<HTMLAudioElement>;

  title = 'ID card detection and Liveness verification';
  isModelLoaded = false;
  isStreaming = false;
  detections: Detection[] = [];
  inferenceTime = 0;
  stream: MediaStream | null = null;
  animationFrameId: number | null = null;
  // Use a relative web path instead of absolute file system path
  modelPath = 'assets/best.onnx';
  errorMessage = '';
  
  // New properties for document verification
  countries: string[] = ['Sri Lanka', 'India', 'United States', 'United Kingdom', 'Australia'];
  selectedCountry: string = 'Sri Lanka';
  documentTypes: string[] = ['ID Card', 'Driving License'];
  selectedDocType: string = 'ID Card';

  // Auto-capture related properties
  highConfidenceStartTime: number = 0;
  highConfidenceTimer: number = 0;
  confidenceThreshold: number = 0.97;
  capturedImage: string | null = null;
  lastFrameTime: number = 0;
  
  // Image quality properties
  originalCapturedImage: string | null = null;
  cropMargin = 20; // Margin in pixels to add around the cropped image
  flashActive: boolean = false;
  captureTimeout: any = null;
  capturingImage: boolean = false;

  constructor(private idCardDetector: IdCardDetectorService) {}

  async ngOnInit(): Promise<void> {
    try {
      // Test if the model file is accessible
      console.log('Testing model access at:', this.modelPath);
      
      try {
        const response = await fetch(this.modelPath);
        if (response.ok) {
          console.log('Model file found! Status:', response.status);
          const arrayBuffer = await response.arrayBuffer();
          console.log('Model size:', arrayBuffer.byteLength, 'bytes');
          
          // Initialize with the fetched model data
          await this.idCardDetector.initialize(arrayBuffer);
          this.isModelLoaded = true;
        } else {
          console.error('Model file not found. Status:', response.status, response.statusText);
          this.errorMessage = `Model file not found: ${response.status} ${response.statusText}`;
        }
      } catch (error) {
        console.error('Error fetching model file:', error);
        this.errorMessage = 'Error fetching model file. Check console for details.';
      }
    } catch (error) {
      console.error('Failed to initialize model:', error);
      this.errorMessage = 'Failed to load ONNX model. Please check console for details.';
    }
  }

  ngOnDestroy(): void {
    this.stopCamera();
    this.clearCaptureTimeout();
  }

  private clearCaptureTimeout(): void {
    if (this.captureTimeout) {
      clearTimeout(this.captureTimeout);
      this.captureTimeout = null;
    }
  }

  async startCamera(): Promise<void> {
    if (!this.isModelLoaded) {
      this.errorMessage = 'Model not loaded yet. Please wait or check for errors.';
      return;
    }

    console.log(`Starting detection for ${this.selectedDocType} from ${this.selectedCountry}`);

    try {
      // Request higher resolution for better quality
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1920 },  // Higher resolution for better quality
          height: { ideal: 1080 },
          facingMode: 'environment' // Use back camera on mobile if available
        }
      });
      
      this.videoElement.nativeElement.srcObject = this.stream;
      this.isStreaming = true;
      this.errorMessage = '';
      
      // Reset auto-capture related properties
      this.highConfidenceStartTime = 0;
      this.highConfidenceTimer = 0;
      this.capturedImage = null;
      this.originalCapturedImage = null;
      this.flashActive = false;
      this.capturingImage = false;
      
      // Wait for video to be ready
      this.videoElement.nativeElement.onloadedmetadata = () => {
        console.log(`Camera initialized with resolution: ${this.videoElement.nativeElement.videoWidth}x${this.videoElement.nativeElement.videoHeight}`);
        this.resizeCanvas();
        this.startDetection();
      };
    } catch (error) {
      console.error('Error accessing camera:', error);
      this.errorMessage = 'Failed to access camera. Please ensure camera permissions are granted.';
    }
  }

  stopCamera(): void {
    this.clearCaptureTimeout();
    
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
    
    this.isStreaming = false;
    this.detections = [];
    this.inferenceTime = 0;
    this.flashActive = false;
  }

  private resizeCanvas(): void {
    const video = this.videoElement.nativeElement;
    const canvas = this.canvasElement.nativeElement;
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }

  private startDetection(): void {
    this.lastFrameTime = performance.now();
    
    const detectFrame = async (currentTime: number) => {
      if (!this.isStreaming || this.capturingImage) return;
      
      // Calculate deltaTime for accurate timing
      const deltaTime = (currentTime - this.lastFrameTime) / 1000; // in seconds
      this.lastFrameTime = currentTime;
      
      const video = this.videoElement.nativeElement;
      const canvas = this.canvasElement.nativeElement;
      const ctx = canvas.getContext('2d');
      
      if (!ctx) return;
      
      // Draw video frame to canvas
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      // Get image data for detection
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      
      try {
        // Run detection
        const result = await this.idCardDetector.detect(imageData);
        this.detections = result.detections;
        this.inferenceTime = result.inferenceTime;
        
        // Draw detections
        this.drawDetections(ctx);
        
        // Check for high confidence detections
        const highConfidenceDetection = this.detections.find(
          det => det.confidence >= this.confidenceThreshold
        );
        
        if (highConfidenceDetection) {
          // If we have a high confidence detection
          if (this.highConfidenceStartTime === 0) {
            // Start the timer if it's not already started
            this.highConfidenceStartTime = currentTime;
          } else {
            // Update the timer
            this.highConfidenceTimer = (currentTime - this.highConfidenceStartTime) / 1000;
            
            // Check if we've had high confidence for long enough (0.5 seconds)
            if (this.highConfidenceTimer >= 0.5) {
              this.capturingImage = true;
              this.captureHighQualityImage(canvas, highConfidenceDetection.box);
              return; // Stop the detection loop
            }
          }
        } else {
          // Reset the timer if confidence drops
          this.highConfidenceStartTime = 0;
          this.highConfidenceTimer = 0;
        }
      } catch (error) {
        console.error('Detection error:', error);
      }
      
      // Request next frame
      this.animationFrameId = requestAnimationFrame(detectFrame);
    };
    
    // Start detection loop
    this.animationFrameId = requestAnimationFrame(detectFrame);
  }

  private drawDetections(ctx: CanvasRenderingContext2D): void {
    const canvas = this.canvasElement.nativeElement;
    
    // Clear previous drawings (we already have the video frame drawn)
    ctx.font = '16px Arial';
    
    // Draw each detection
    for (const det of this.detections) {
      const [x1, y1, x2, y2] = det.box;
      const confidence = det.confidence;
      const label = `${det.class_name}: ${(confidence * 100).toFixed(1)}%`;
      
      // Draw bounding box with different color for high confidence
      ctx.strokeStyle = confidence >= this.confidenceThreshold ? 'rgb(255, 215, 0)' : 'rgb(0, 255, 0)';
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      
      // Draw label background
      const textMeasure = ctx.measureText(label);
      const textHeight = 20;
      ctx.fillStyle = confidence >= this.confidenceThreshold ? 'rgb(255, 215, 0)' : 'rgb(0, 255, 0)';
      ctx.fillRect(x1, y1 - textHeight, textMeasure.width + 10, textHeight);
      
      // Draw label text
      ctx.fillStyle = 'rgb(0, 0, 0)';
      ctx.fillText(label, x1 + 5, y1 - 5);
    }
    
    // Draw inference time
    ctx.fillStyle = 'rgb(255, 0, 0)';
    ctx.font = '20px Arial';
    ctx.fillText(`Inference time: ${this.inferenceTime.toFixed(1)} ms`, 10, 30);
    
    // Draw auto-capture timer if active
    if (this.highConfidenceTimer > 0) {
      ctx.fillStyle = 'rgb(255, 215, 0)';
      ctx.font = '20px Arial';
      ctx.fillText(`Auto-capture in: ${(0.5 - this.highConfidenceTimer).toFixed(1)}s`, 10, 60);
    }
  }
  
  private captureHighQualityImage(canvas: HTMLCanvasElement, box: number[]): void {
    // Clear any existing timeout
    this.clearCaptureTimeout();
    
    // 1. Activate flash
    this.flashActive = true;
    
    // 2. Try to focus the video element
    if (this.videoElement && this.videoElement.nativeElement) {
      try {
        // Call focus if available
        if (typeof this.videoElement.nativeElement.focus === 'function') {
          this.videoElement.nativeElement.focus();
        }
      } catch (err) {
        console.warn('Could not focus video element:', err);
      }
    }
    
    // 3. Wait a moment for the camera to stabilize and adjust
    this.captureTimeout = setTimeout(() => {
      try {
        // Play camera shutter sound if available
        if (this.shutterSound && this.shutterSound.nativeElement) {
          this.shutterSound.nativeElement.play().catch(err => {
            console.warn('Could not play shutter sound:', err);
          });
        }
        
        // Get the latest video frame
        const video = this.videoElement.nativeElement;
        const ctx = canvas.getContext('2d');
        
        if (ctx) {
          // Redraw the video to canvas to get the latest frame
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          
          // Store full image at maximum quality
          this.originalCapturedImage = canvas.toDataURL('image/png', 1.0);
          
          // Now crop to the card area
          this.capturedImage = this.cropImage(canvas, box);
          
          console.log('High quality ID card image captured!');
        }
      } catch (error) {
        console.error('Error capturing high quality image:', error);
      } finally {
        // Turn off flash and clean up
        this.flashActive = false;
        this.capturingImage = false;
        this.stopCamera();
      }
    }, 500); // 500ms delay to stabilize
  }
  
  /**
   * Crop the image to focus on the ID card with high quality
   */
  private cropImage(canvas: HTMLCanvasElement, box: number[]): string {
    const [x1, y1, x2, y2] = box;
    let width = x2 - x1;
    let height = y2 - y1;
    
    // Add margin to the cropped area
    const x = Math.max(0, x1 - this.cropMargin);
    const y = Math.max(0, y1 - this.cropMargin);
    width = Math.min(canvas.width - x, width + this.cropMargin * 2);
    height = Math.min(canvas.height - y, height + this.cropMargin * 2);
    
    // Create a temporary canvas for the cropped image
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = width;
    tempCanvas.height = height;
    
    const tempCtx = tempCanvas.getContext('2d');
    if (!tempCtx) return '';
    
    // Apply a slight sharpening filter to enhance details
    tempCtx.filter = 'contrast(1.1) saturate(1.1)';
    
    // Draw the cropped part to the temporary canvas
    tempCtx.drawImage(
      canvas, 
      x, y, width, height,  // Source rectangle
      0, 0, width, height   // Destination rectangle
    );
    
    // Reset filter
    tempCtx.filter = 'none';
    
    // Get the image data as PNG (lossless) for maximum quality
    return tempCanvas.toDataURL('image/png', 1.0);
  }

  manualCapture(): void {
    if (!this.isStreaming || this.capturingImage) return;
    
    const canvas = this.canvasElement.nativeElement;
    
    // Find high confidence detection if available
    const highConfidenceDetection = this.detections.find(
      det => det.confidence >= this.confidenceThreshold
    );
    
    if (highConfidenceDetection) {
      this.capturingImage = true;
      this.captureHighQualityImage(canvas, highConfidenceDetection.box);
    } else if (this.detections.length > 0) {
      // Use the highest confidence detection available
      const bestDetection = [...this.detections].sort((a, b) => b.confidence - a.confidence)[0];
      this.capturingImage = true;
      this.captureHighQualityImage(canvas, bestDetection.box);
    } else {
      alert('No ID card detected. Please position your ID card within view.');
    }
  }

  retakePhoto(): void {
    this.capturedImage = null;
    this.originalCapturedImage = null;
    this.startCamera();
  }
  
  proceedWithImage(): void {
    console.log('Proceeding with the captured image...');
    // Logic for proceeding will be added later
    alert('Proceeding with the captured image!');
  }
  
  // For debugging - toggle between cropped and original image
  toggleImageView(): void {
    if (this.originalCapturedImage && this.capturedImage) {
      const temp = this.capturedImage;
      this.capturedImage = this.originalCapturedImage;
      this.originalCapturedImage = temp;
    }
  }
}