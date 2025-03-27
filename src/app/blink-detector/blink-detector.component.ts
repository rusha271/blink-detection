import { Component, ElementRef, OnInit, OnDestroy, ViewChild, PLATFORM_ID, Inject, AfterViewInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule, isPlatformBrowser } from '@angular/common';
import { HttpClient, HttpClientModule, HttpErrorResponse } from '@angular/common/http';
import { BehaviorSubject, Subject } from 'rxjs';
import * as faceapi from 'face-api.js';

// Interfaces to define structured data types used throughout the component
interface DetectionState {
  isBlinking: boolean;           // Indicates if a blink is currently being detected based on eye aspect ratio
  lastBlinkTime: number;         // Stores the timestamp (in milliseconds) of the last detected blink to enforce minimum intervals
  blinkCount: number;            // Keeps a running total of detected blinks during the test
  currentEyeDistance: number;    // Represents the current Eye Aspect Ratio (EAR), a measure of eye openness
  baselineEyeDistance: number;   // The average EAR calculated during calibration, used as a reference for blink detection
  calibrationFrames: number[];   // Array storing EAR values collected during calibration to establish the baseline
  lastLandmarks?: faceapi.FaceLandmarks68; // Stores the most recent facial landmarks detected by face-api.js
  motionDetected: boolean;       // Flags whether significant motion has been detected to differentiate live video from static images
  stage: 'Scanning' | 'waiting' | 'blinking' | 'headMovement' | 'complete'; // Tracks the current phase of the liveness test
  requiredMovements: string[];   // List of head movements the user must perform (e.g., 'left', 'right')
  completedMovements: string[];  // List of head movements the user has successfully completed
  currentMovementDirection: string | null; // The direction of the current head movement detected (e.g., 'up', 'down', or null if none)
}

interface CapturedImage {
  id: string;                    // A unique identifier for each captured image, generated using timestamp and random string
  base64: string;                // The image data encoded in base64 format for storage and transmission
  timestamp: number;             // The exact time (in milliseconds) when the image was captured
  event: 'blink' | 'headMovement'; // Specifies the type of event that triggered the capture (blink or head movement)
  movement?: string;             // Optional: specifies the direction of head movement (e.g., 'left') if the event is 'headMovement'
  matchPercentage?: number;      // Optional: percentage match score returned by the backend for this image
}

interface BackendResponse {
  message: string;               // A human-readable message from the backend describing the result
  match: boolean;                // Indicates if the captured data matched the expected criteria (e.g., face verification)
  averageMatchPercentage: number; // The average match percentage across all submitted images
  details: {                     // Provides detailed information about the backend's analysis
    receivedImagesCount: number; // The number of images the backend received
    capturedData: Array<{
      event: string;             // The event type ('blink' or 'headMovement') for each analyzed image
      movement?: string;         // The movement direction, if applicable
      matchPercentage: number;   // The match percentage for this specific image
      timestamp?: number;        // The timestamp of the image, if provided
    }>;
  };
}

@Component({
  selector: 'app-blink-detector',
  templateUrl: './blink-detector.component.html',
  styleUrls: ['./blink-detector.component.css'],
  standalone: true,
  imports: [CommonModule, HttpClientModule],
})
export class BlinkDetectorComponent implements OnInit, AfterViewInit, OnDestroy {
  // References to the video and canvas elements in the template, used for displaying the camera feed and drawing overlays
  @ViewChild('videoElement') private videoElement!: ElementRef<HTMLVideoElement>;
  @ViewChild('canvasElement') private canvasElement!: ElementRef<HTMLCanvasElement>;

  // Observables and state variables to manage the component's data and UI
  public currentTime$ = new BehaviorSubject<number>(0); // BehaviorSubject to emit the current elapsed time in seconds
  public capturedImages: CapturedImage[] = [];          // Array storing all images captured during the test
  public selectedImage: CapturedImage | null = null;   // The currently selected image for viewing in the UI
  public detectionState: DetectionState = {            // Initial state for tracking detection progress
    isBlinking: false,                                // No blink detected initially
    lastBlinkTime: 0,                                 // No previous blink
    blinkCount: 0,                                    // No blinks counted yet
    currentEyeDistance: 0,                            // Current EAR starts at 0
    baselineEyeDistance: 0,                           // Baseline EAR starts at 0, updated during calibration
    calibrationFrames: [],                            // Empty array for calibration data
    motionDetected: false,                            // No motion detected initially
    stage: 'Scanning',                                // Starts in the 'Scanning' stage
    requiredMovements: [],                            // No required movements yet
    completedMovements: [],                           // No completed movements yet
    currentMovementDirection: null,                   // No current movement direction
  };

  // UI-related state variables to control the display and user interaction
  public isLoading = true;           // Indicates that the component is loading (e.g., models or camera setup)
  public error: string | null = null; // Stores any error message to display to the user
  public detectionActive = false;    // Tracks whether the detection process is currently running
  public isCalibrating = false;      // Indicates if the calibration phase is active
  public calibrationProgress = 0;    // Percentage of calibration completion (0-100)
  public distanceMessage = '';       // Message to guide the user about face alignment or lighting
  public finalResult: string | null = null; // Final result message after the test completes
  public results: Record<number, string> = {}; // Stores blink detection results per second (e.g., { 1: 'yes' })
  public movementInstruction = '';   // Instructions to display to the user (e.g., "Blink now")
  public timeRemaining: number = 0;  // Remaining time in seconds for the current stage
  public matchDetails: BackendResponse | null = null; // Stores the backend's response after image submission
  public showRetryButton = false;    // Controls visibility of the retry button
  public showGoAheadButton = false;  // Controls visibility of the go-ahead button
  public maxAttemptsReached = false; // Flags if the maximum number of attempts has been reached
  public showStartButton = true;     // Controls visibility of the start button, hidden after first attempt
  private attemptCount = 0;          // Tracks the number of detection attempts made
  private isSafariOrIOS: boolean = false; // Detects if the browser is Safari or the device is iOS for compatibility fixes

  // Constants defining the behavior and thresholds of the detection process
  private readonly CALIBRATION_FRAMES = 150;        // Number of frames to collect during calibration to establish baseline EAR
  private readonly TOTAL_TEST_DURATION = 30000;     // Total duration of the liveness test in milliseconds (30 seconds)
  private readonly BLINKING_DURATION = 10000;       // Duration of the blinking stage in milliseconds (10 seconds)
  private readonly HEAD_MOVEMENT_DURATION = 10000;  // Duration of the head movement stage in milliseconds (10 seconds)
  private readonly WAITING_DURATION = 5000;         // Duration of waiting stages between phases in milliseconds (5 seconds)
  private readonly MOTION_THRESHOLD = 0.02;         // Minimum motion value to detect live video (prevents static images)
  private readonly STATIC_FRAME_THRESHOLD = 180;    // Number of frames without motion to flag as static (3 seconds at 60fps)
  private readonly MOTION_BUFFER_SIZE = 30;         // Size of the buffer to track recent motion frames
  private readonly MIN_MOTION_COUNT = 5;            // Minimum number of motion frames in the buffer to confirm live video
  private readonly HEAD_MOVEMENT_THRESHOLD = 30;    // Pixel threshold for detecting head movement (e.g., left, right)
  private readonly POSSIBLE_MOVEMENTS = ['left', 'right', 'up', 'down']; // Array of possible head movement directions
  private readonly MATCH_THRESHOLD = 50;            // Minimum average match percentage from the backend for success
  private readonly MAX_ATTEMPTS = 3;                // Maximum number of attempts allowed before locking out
  private readonly NOSE_MOTION_THRESHOLD = 3;       // Maximum nose motion (in pixels) allowed during blink detection to ensure stability

  private readonly BLINK_SETTINGS = {               // Configuration for blink detection
    EYE_ASPECT_RATIO_THRESHOLD: 0.23,              // Initial threshold for EAR to detect a blink (adjusted during calibration)
    MIN_BLINK_FRAMES: 2,                           // Minimum consecutive frames below threshold to count as a blink
    MAX_BLINK_FRAMES: 10,                          // Maximum frames for a blink to avoid false positives (e.g., eyes closed too long)
    MIN_BLINK_INTERVAL: 500,                       // Minimum time (ms) between blinks to avoid counting rapid blinks as one
  };

  // Private variables for managing internal state and resources
  private readonly destroy$ = new Subject<void>();  // Subject to signal component destruction and clean up subscriptions
  private startTime = 0;                            // Timestamp when the detection starts
  private frameCount = 0;                           // Total number of frames processed during detection
  private consecutiveBlinkFrames = 0;               // Number of consecutive frames where EAR indicates a blink
  private readonly eyeDistanceWindow: number[] = []; // Rolling window of recent EAR values (not currently used but reserved)
  private staticFrameCounter = 0;                   // Counts frames with no motion to detect static images
  private motionBuffer: boolean[] = [];             // Buffer storing recent motion detection results (true/false)
  private motionCount = 0;                          // Number of motion frames in the buffer
  private animationFrameId?: number;                // ID of the requestAnimationFrame for face detection loop
  private baselineHeadPosition = { x: 0, y: 0, z: 0 }; // Baseline head position (nose and eye distance) for movement detection
  private stageTimer: any = null;                   // Timer ID for managing stage transitions
  private scanningSpeed = 0.005;                    // Speed of the scanning animation during calibration

  private mediaRecorder: MediaRecorder | null = null; // MediaRecorder instance for video recording
  private recordedChunks: Blob[] = [];              // Array of video data chunks recorded
  public recordedVideoBase64: string | null = null; // Base64-encoded video data after recording

  private storedCalibration: { baselineEyeDistance: number; eyeAspectRatioThreshold: number } | null = null; // Persists calibration data across retries

  // Constructor with dependency injection
  constructor(
    @Inject(PLATFORM_ID) private platformId: Object, // Injected platform ID to determine if running in a browser
    private http: HttpClient,                        // HTTP client for sending data to the backend
    private cdr: ChangeDetectorRef                   // ChangeDetectorRef for manually triggering change detection
  ) {}

  // Lifecycle hook: Runs when the component is initialized
  async ngOnInit() {
    // Ensure the component is running in a browser environment (not server-side rendering)
    if (!isPlatformBrowser(this.platformId)) {
      this.handleError('Camera features unavailable outside browser.');
      return;
    }

    // Detect if the browser is Safari or the device is iOS, as these require special handling for video playback
    const ua = navigator.userAgent;
    this.isSafariOrIOS = /Safari/i.test(ua) && !/Chrome/i.test(ua) || /iPhone|iPad|iPod/i.test(ua);
    console.debug('Browser detection - Safari/iOS:', this.isSafariOrIOS);

    // Check for HTTPS protocol (required for camera access), allowing localhost as an exception
    if (window.location.protocol !== 'https:' && window.location.hostname !== 'localhost') {
      this.handleError('Camera access requires HTTPS. Please use a secure connection.');
      return;
    }

    // Load the face-api.js models needed for face detection and landmark recognition
    await this.loadFaceApiModels();
  }

  // Lifecycle hook: Runs after the component's view is initialized
  async ngAfterViewInit() {
    // Skip if not in a browser environment
    if (!isPlatformBrowser(this.platformId)) return;

    // Poll until loading is complete and DOM elements are available
    const checkElements = setInterval(async () => {
      if (!this.isLoading) {
        clearInterval(checkElements); // Stop polling once loading is done
        // Verify that video and canvas elements are present
        if (!this.videoElement || !this.canvasElement) {
          this.handleError('Video or canvas element not found in the template after loading.');
          return;
        }
        // Initialize the detector with camera and model setup
        await this.initializeDetector();
      }
    }, 100); // Check every 100ms
  }

  // Loads face-api.js models from the assets folder
  private async loadFaceApiModels() {
    try {
      this.isLoading = true; // Show loading state
      this.cdr.detectChanges(); // Update UI
      console.debug('Loading face-api.js models...');
      // Load the Tiny Face Detector and Face Landmark models asynchronously
      await faceapi.loadTinyFaceDetectorModel('/assets/models');
      await faceapi.loadFaceLandmarkModel('/assets/models');
      this.isLoading = false; // Hide loading state
      console.debug('Models loaded successfully.');
      this.cdr.detectChanges(); // Update UI
    } catch (error) {
      // Handle any errors during model loading
      this.handleError(`Model loading failed: ${(error as Error).message}`);
    }
  }

  // Initializes the detector by loading models and setting up the camera
  private async initializeDetector() {
    try {
      console.debug('Initializing detector...');
      // Ensure models are loaded (redundant but ensures availability)
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri('/assets/models'),
        faceapi.nets.faceLandmark68Net.loadFromUri('/assets/models'),
      ]);

      // Check that video and canvas elements are available
      if (!this.videoElement || !this.videoElement.nativeElement || !this.canvasElement || !this.canvasElement.nativeElement) {
        throw new Error('Video or canvas element is not properly initialized.');
      }

      // Set up the camera stream
      await this.setupCamera();

      // Match canvas dimensions to the video stream
      this.canvasElement.nativeElement.width = this.videoElement.nativeElement.videoWidth;
      this.canvasElement.nativeElement.height = this.videoElement.nativeElement.videoHeight;
      console.debug('Detector initialized successfully.');
      this.cdr.detectChanges(); // Update UI
    } catch (error) {
      // Handle initialization errors
      this.handleError(`Initialization failed: ${(error as Error).message}`);
    }
  }

  // Sets up the camera stream using the front-facing camera
  private async setupCamera() {
    try {
      console.debug('Setting up camera with user (front) facing mode...');
      // Request access to the user's front-facing camera
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user' }, // Prefer front camera
      });

      // Ensure video element exists
      if (!this.videoElement?.nativeElement) {
        throw new Error('Video element is not initialized.');
      }

      // Assign the stream to the video element
      this.videoElement.nativeElement.srcObject = stream;

      // Handle Safari/iOS-specific playback issues
      if (this.isSafariOrIOS) {
        console.debug('Applying Safari/iOS autoplay fix...');
        this.videoElement.nativeElement.muted = true; // Mute to allow autoplay
        this.videoElement.nativeElement.play().catch((err) => {
          console.error('Safari/iOS play failed:', err);
          this.handleError('Video playback failed on Safari/iOS. Please allow camera and refresh.');
        });
      }

      // Wait for video metadata to load and start playback
      await new Promise<void>((resolve, reject) => {
        this.videoElement!.nativeElement.onloadedmetadata = () => {
          console.debug('Video metadata loaded. Playing video...');
          this.videoElement!.nativeElement.play().then(resolve).catch(reject);
        };
        this.videoElement!.nativeElement.onerror = () => reject(new Error('Video element error during setup.'));
      });

      console.debug('Camera setup complete.');
      this.cdr.detectChanges(); // Update UI
    } catch (error) {
      // Handle camera access or playback errors
      this.handleError(`Camera access failed: ${(error as Error).message}`);
    }
  }

  // Stops the camera stream and cleans up resources
  private stopCameraStream() {
    // Check if in a browser and if a stream exists
    if (isPlatformBrowser(this.platformId) && this.videoElement?.nativeElement?.srcObject) {
      const stream = this.videoElement.nativeElement.srcObject as MediaStream;
      // Stop all video tracks
      stream.getTracks().forEach((track) => track.stop());
      this.videoElement.nativeElement.srcObject = null; // Clear the stream
      console.debug('Camera stream stopped.');
    }
  }

  // Starts recording video from the camera stream
  private startVideoRecording() {
    if (!this.videoElement?.nativeElement) return; // Exit if video element is missing
    const stream = this.videoElement.nativeElement.srcObject as MediaStream;
    if (!stream) {
      console.error('No stream available for recording.');
      return;
    }

    this.recordedChunks = []; // Reset recorded chunks
    try {
      // Initialize MediaRecorder with WebM format
      this.mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
      console.debug('Starting video recording...');

      // Collect video data as it becomes available
      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) this.recordedChunks.push(event.data);
      };

      // Handle recording stop and convert to base64
      this.mediaRecorder.onstop = () => {
        console.debug('Recording stopped. Compiling video...');
        const blob = new Blob(this.recordedChunks, { type: 'video/webm' });
        this.convertBlobToBase64(blob).then((base64Video) => {
          this.recordedVideoBase64 = base64Video; // Store the base64 video
          console.debug('Recorded video base64 length:', this.recordedVideoBase64.length);
          this.cdr.detectChanges(); // Update UI
        }).catch((err) => {
          console.error('Failed to convert video to base64:', err);
          this.handleError('Video encoding failed.');
        });
      };

      // Handle recording errors
      this.mediaRecorder.onerror = (event) => {
        console.error('MediaRecorder error:', event);
        this.handleError('Video recording error occurred.');
      };

      this.mediaRecorder.start(); // Begin recording
      // Automatically stop after 15 seconds
      setTimeout(() => {
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
          this.mediaRecorder.stop();
          console.debug('Recording stopped after 15 seconds.');
        }
      }, 15000);
    } catch (error) {
      console.error('Failed to start video recording:', error);
      this.handleError(`Video recording setup failed: ${(error as Error).message}`);
    }
  }

  // Converts a Blob to a base64 string
  private convertBlobToBase64(blob: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result as string); // Resolve with base64 data
      reader.onerror = reject; // Reject on error
      reader.readAsDataURL(blob); // Read the blob as a data URL
    });
  }

  // Public method to start the detection process
  public async startDetection(): Promise<void> {
    // Check if maximum attempts have been reached
    if (this.attemptCount >= this.MAX_ATTEMPTS) {
      this.maxAttemptsReached = true;
      this.finalResult = 'Maximum attempts reached. Please contact support.';
      this.cdr.detectChanges();
      return;
    }
    this.attemptCount++; // Increment attempt counter
    this.showStartButton = false; // Hide start button after first attempt

    console.debug('Starting detection...');
    this.resetState(); // Reset all state variables to initial values

    this.startFaceDetection(); // Begin the face detection loop

    // Use stored calibration data if available (after first attempt)
    if (this.attemptCount > 1 && this.storedCalibration) {
      console.debug('Using stored calibration data...');
      this.detectionState.baselineEyeDistance = this.storedCalibration.baselineEyeDistance;
      this.BLINK_SETTINGS.EYE_ASPECT_RATIO_THRESHOLD = this.storedCalibration.eyeAspectRatioThreshold;
    } else {
      // Perform calibration on the first attempt
      await this.calibrate();
      this.storedCalibration = {
        baselineEyeDistance: this.detectionState.baselineEyeDistance,
        eyeAspectRatioThreshold: this.BLINK_SETTINGS.EYE_ASPECT_RATIO_THRESHOLD,
      };
      console.debug('Calibration stored:', this.storedCalibration);
    }

    this.startTime = Date.now(); // Record start time
    this.detectionActive = true; // Activate detection
    this.detectionState.stage = 'waiting'; // Start with waiting stage
    this.startVideoRecording(); // Begin video recording
    this.manageStages(); // Start stage transitions

    // Update elapsed time every 100ms
    const timeUpdateInterval = setInterval(() => {
      const elapsed = Date.now() - this.startTime;
      this.currentTime$.next(elapsed / 1000); // Emit time in seconds

      // End test if total duration is reached
      if (elapsed >= this.TOTAL_TEST_DURATION) {
        clearInterval(timeUpdateInterval);
        this.completeTest();
      }
    }, 100);

    this.cdr.detectChanges(); // Update UI
  }

  // Captures an image from the video stream
  private captureImage(event: 'blink' | 'headMovement', movement?: string): void {
    if (!this.videoElement?.nativeElement || !this.canvasElement?.nativeElement) return; // Exit if elements are missing

    // Create a temporary canvas to draw the current video frame
    const canvas = document.createElement('canvas');
    canvas.width = this.videoElement.nativeElement.videoWidth;
    canvas.height = this.videoElement.nativeElement.videoHeight;
    const ctx = canvas.getContext('2d');

    if (!ctx) {
      console.error('Failed to get 2D context for image capture.');
      return;
    }

    // Draw the current video frame onto the canvas
    ctx.drawImage(this.videoElement.nativeElement, 0, 0);
    const base64Data = canvas.toDataURL('image/jpeg', 0.8); // Convert to base64 with 80% quality
    const id = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`; // Generate unique ID

    // Create captured image object
    const capturedImage: CapturedImage = {
      id,
      base64: base64Data,
      timestamp: Date.now(),
      event,
      movement,
      matchPercentage: undefined, // Match percentage will be set later by backend
    };

    this.capturedImages.push(capturedImage); // Add to captured images array
    console.debug(`Captured image for ${event}${movement ? ' - ' + movement : ''}`);
    this.cdr.detectChanges(); // Update UI
  }

  // Manages the sequence of stages in the liveness test
  private manageStages(): void {
    this.updateTimeRemaining(this.WAITING_DURATION / 1000); // Set initial time remaining
    this.movementInstruction = 'Prepare for blinking test...'; // Initial instruction
    this.cdr.detectChanges();

    // Transition to blinking stage after waiting
    this.stageTimer = setTimeout(() => {
      if (this.detectionState.lastLandmarks) {
        this.baselineHeadPosition = this.calculateHeadPosition(this.detectionState.lastLandmarks); // Set baseline for head movement
      }
      this.detectionState.stage = 'blinking';
      this.movementInstruction = 'Please blink naturally several times';
      this.updateTimeRemaining(this.BLINKING_DURATION / 1000);
      this.cdr.detectChanges();

      // Transition to waiting before head movement
      this.stageTimer = setTimeout(() => {
        this.detectionState.stage = 'waiting';
        this.movementInstruction = 'Prepare for head movement test...';
        this.updateTimeRemaining(this.WAITING_DURATION / 1000);
        this.cdr.detectChanges();

        // Transition to head movement stage
        this.stageTimer = setTimeout(() => {
          this.detectionState.stage = 'headMovement';
          this.initializeHeadMovementCheck(); // Set up required movements
          this.updateTimeRemaining(this.HEAD_MOVEMENT_DURATION * 2 / 1000); // Double duration for two movements
          this.cdr.detectChanges();

          // End test after head movement duration
          this.stageTimer = setTimeout(() => {
            this.completeTest();
          }, this.HEAD_MOVEMENT_DURATION * 2);
        }, this.WAITING_DURATION);
      }, this.BLINKING_DURATION);
    }, this.WAITING_DURATION);
  }

  // Updates the time remaining countdown for the current stage
  private updateTimeRemaining(seconds: number): void {
    this.timeRemaining = seconds; // Set initial time
    const countdownInterval = setInterval(() => {
      this.timeRemaining--; // Decrement every second
      if (this.timeRemaining <= 0) clearInterval(countdownInterval); // Stop when time runs out
      this.cdr.detectChanges(); // Update UI
    }, 1000); // Update every 1 second
  }

  // Calculates the average brightness of the current video frame
  private calculateFrameBrightness(video: HTMLVideoElement): number {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return 0; // Return 0 if context is unavailable

    ctx.drawImage(video, 0, 0); // Draw video frame onto canvas
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data; // Get pixel data

    let sum = 0;
    // Calculate luminance for each pixel using RGB weights
    for (let i = 0; i < data.length; i += 4) {
      const luminance = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      sum += luminance;
    }
    return sum / (data.length / 4); // Return average brightness
  }

  // Starts the continuous face detection loop
  private startFaceDetection() {
    let lastTime = Date.now(); // Track time between frames for debugging
    const detectFace = async () => {
      try {
        if (!this.videoElement?.nativeElement) return; // Exit if video element is missing

        const currentTime = Date.now();
        console.debug(`Time between frames: ${currentTime - lastTime}ms`);
        lastTime = currentTime;

        // Check frame brightness for lighting conditions
        const brightness = this.calculateFrameBrightness(this.videoElement.nativeElement);
        if (brightness < 50) {
          this.distanceMessage = 'Lighting too dim. Please increase brightness.';
        } else if (brightness > 200) {
          this.distanceMessage = 'Lighting too bright. Please reduce brightness.';
        } else {
          this.distanceMessage = ''; // Clear message if lighting is acceptable
        }
        this.cdr.detectChanges();

        // Detect faces and landmarks in the current frame
        const detections = await faceapi
          .detectAllFaces(this.videoElement.nativeElement, new faceapi.TinyFaceDetectorOptions())
          .withFaceLandmarks();

        if (detections.length > 0) {
          const landmarks = detections[0].landmarks; // Use the first detected face
          await this.handleFaceDetectionResults(landmarks); // Process the landmarks
        } else if (!this.isCalibrating && !this.distanceMessage) {
          // Show message if no face is detected and not calibrating
          this.distanceMessage = 'No face detected. Please align your face with the camera.';
          this.cdr.detectChanges();
        }

        // Continue the loop if detection or calibration is active
        if (this.detectionActive || this.isCalibrating) {
          this.animationFrameId = requestAnimationFrame(detectFace); // Schedule next frame
        }
      } catch (error) {
        console.error('Detection error:', error);
      }
    };

    detectFace(); // Start the detection loop
  }

  // Initializes the head movement check by selecting required movements
  private initializeHeadMovementCheck(): void {
    // Randomly shuffle and select two movements
    const shuffled = [...this.POSSIBLE_MOVEMENTS].sort(() => 0.5 - Math.random());
    this.detectionState.requiredMovements = shuffled.slice(0, 2);
    this.movementInstruction = `Please move your head ${this.detectionState.requiredMovements[0]}, then ${this.detectionState.requiredMovements[1]}`;
    // Set baseline head position if landmarks are available
    this.baselineHeadPosition = this.calculateHeadPosition(this.detectionState.lastLandmarks!);
    this.cdr.detectChanges();
  }

  // Calculates the head position based on nose and eye landmarks
  private calculateHeadPosition(landmarks: faceapi.FaceLandmarks68): { x: number; y: number; z: number } {
    const nose = landmarks.getNose()[0]; // Nose tip position
    const leftEye = landmarks.getLeftEye()[0]; // Left eye corner
    const rightEye = landmarks.getRightEye()[3]; // Right eye corner

    return {
      x: nose.x, // Horizontal position
      y: nose.y, // Vertical position
      z: this.euclideanDistance(leftEye, rightEye), // Distance between eyes (depth proxy)
    };
  }

  // Calibrates the baseline EAR by collecting frames
  private async calibrate(): Promise<void> {
    this.isCalibrating = true; // Start calibration
    this.calibrationProgress = 0; // Reset progress
    this.detectionState.calibrationFrames = []; // Clear previous frames
    this.cdr.detectChanges();

    return new Promise<void>((resolve) => {
      const calibrationInterval = setInterval(() => {
        // Check if enough frames have been collected
        if (this.detectionState.calibrationFrames.length >= this.CALIBRATION_FRAMES) {
          clearInterval(calibrationInterval);
          this.detectionState.baselineEyeDistance = this.calculateBaselineEyeDistance(); // Calculate baseline
          this.BLINK_SETTINGS.EYE_ASPECT_RATIO_THRESHOLD = this.detectionState.baselineEyeDistance * 0.9; // Set threshold to 90% of baseline
          console.debug(`Calibrated EAR threshold: ${this.BLINK_SETTINGS.EYE_ASPECT_RATIO_THRESHOLD.toFixed(3)}`);
          this.isCalibrating = false; // End calibration
          this.cdr.detectChanges();
          resolve(); // Resolve the promise
        }
        // Update progress percentage
        this.calibrationProgress = (this.detectionState.calibrationFrames.length / this.CALIBRATION_FRAMES) * 100;
        this.cdr.detectChanges();
      }, 100); // Check every 100ms
    });
  }

  // Calculates the median EAR from calibration frames
  private calculateBaselineEyeDistance(): number {
    const sortedDistances = [...this.detectionState.calibrationFrames].sort((a, b) => a - b); // Sort frames
    const medianIndex = Math.floor(sortedDistances.length / 2); // Find median index
    return sortedDistances[medianIndex]; // Return median value
  }

  // Calculates the motion of the nose between frames
  private calculateNoseMotion(currentLandmarks: faceapi.FaceLandmarks68, previousLandmarks: faceapi.FaceLandmarks68): number {
    const currentNose = currentLandmarks.getNose()[0];
    const previousNose = previousLandmarks.getNose()[0];
    const dx = currentNose.x - previousNose.x; // Horizontal difference
    const dy = currentNose.y - previousNose.y; // Vertical difference
    return Math.sqrt(dx * dx + dy * dy); // Euclidean distance
  }

  // Processes face detection results (landmarks) for calibration or detection
  private async handleFaceDetectionResults(landmarks: faceapi.FaceLandmarks68): Promise<void> {
    const eyeAR = this.calculateEyeAspectRatio(landmarks); // Calculate current EAR

    // During calibration, collect EAR values
    if (this.isCalibrating) {
      this.detectionState.calibrationFrames.push(eyeAR);
      this.drawResults(landmarks); // Draw calibration overlay
      return;
    }

    let noseMotion = 0;
    // Check for motion if previous landmarks exist
    if (this.detectionState.lastLandmarks) {
      const motion = this.calculateMotion(landmarks, this.detectionState.lastLandmarks); // Overall landmark motion
      const hasMotion = motion > this.MOTION_THRESHOLD; // Check if motion exceeds threshold

      // Update motion buffer
      this.motionBuffer.push(hasMotion);
      if (this.motionBuffer.length > this.MOTION_BUFFER_SIZE) {
        const removedMotion = this.motionBuffer.shift(); // Remove oldest entry
        if (removedMotion) this.motionCount--; // Decrement motion count if removed was true
      }
      if (hasMotion) this.motionCount++; // Increment motion count if motion detected

      // Confirm live video if enough motion is detected
      if (this.motionCount >= this.MIN_MOTION_COUNT) {
        this.detectionState.motionDetected = true;
        this.staticFrameCounter = 0; // Reset static counter
      } else {
        this.staticFrameCounter++; // Increment static counter
        // Flag as static if no motion for too long
        if (this.staticFrameCounter > this.STATIC_FRAME_THRESHOLD) {
          this.handleError('Static image detected. Please use a live video.');
          return;
        }
      }
      noseMotion = this.calculateNoseMotion(landmarks, this.detectionState.lastLandmarks); // Calculate nose-specific motion
    }
    this.detectionState.lastLandmarks = landmarks; // Store current landmarks

    if (!this.detectionActive) return; // Exit if detection is not active

    // Handle blinking stage
    if (this.detectionState.stage === 'blinking') {
      this.updateBlinkDetection(eyeAR, noseMotion);
    } 
    // Handle head movement stage
    else if (this.detectionState.stage === 'headMovement') {
      const currentPosition = this.calculateHeadPosition(landmarks);
      this.checkHeadMovement(currentPosition); // Check for movement
    }

    this.drawResults(landmarks); // Draw detection overlay
  }

  // Checks for head movement based on current position
  private checkHeadMovement(currentPosition: { x: number; y: number; z: number }): void {
    // Exit if all required movements are completed
    if (this.detectionState.completedMovements.length >= 2) return;

    const movement = this.detectMovement(currentPosition); // Detect movement direction

    this.detectionState.currentMovementDirection = movement; // Update current direction

    // Process detected movement
    if (movement && !this.detectionState.completedMovements.includes(movement)) {
      const expectedMovement = this.detectionState.requiredMovements[this.detectionState.completedMovements.length];
      if (movement === expectedMovement) { // Check if it matches the next required movement
        this.captureImage('headMovement', movement); // Capture the image
        this.detectionState.completedMovements.push(movement); // Mark as completed
        // Update instruction if more movements are needed
        if (this.detectionState.completedMovements.length < 2) {
          this.movementInstruction = `Good! Now move your head ${this.detectionState.requiredMovements[1]}`;
          this.updateTimeRemaining(this.HEAD_MOVEMENT_DURATION / 1000);
        }
        this.cdr.detectChanges();
      }
    }
  }

  // Allows viewing a captured image by ID
  public viewCapturedImage(imageId: string): void {
    this.selectedImage = this.capturedImages.find((img) => img.id === imageId) || null; // Find and set selected image
    this.cdr.detectChanges();
  }

  // Detects head movement direction based on position change
  private detectMovement(currentPosition: { x: number; y: number; z: number }): string | null {
    const dx = currentPosition.x - this.baselineHeadPosition.x; // Horizontal change
    const dy = currentPosition.y - this.baselineHeadPosition.y; // Vertical change

    // Check horizontal movement
    if (Math.abs(dx) > this.HEAD_MOVEMENT_THRESHOLD) {
      return dx > 0 ? 'right' : 'left';
    }
    // Check vertical movement
    if (Math.abs(dy) > this.HEAD_MOVEMENT_THRESHOLD) {
      return dy > 0 ? 'down' : 'up';
    }
    return null; // No significant movement
  }

  // Calculates the Eye Aspect Ratio (EAR) for blink detection
  private calculateEyeAspectRatio(landmarks: faceapi.FaceLandmarks68): number {
    const getEyeAR = (eye: faceapi.Point[]) => {
      if (eye.length < 6) return 0; // Return 0 if not enough points
      const v1 = this.euclideanDistance(eye[1], eye[5]); // Vertical distance 1
      const v2 = this.euclideanDistance(eye[2], eye[4]); // Vertical distance 2
      const h = this.euclideanDistance(eye[0], eye[3]);  // Horizontal distance
      return (v1 + v2) / (2.0 * h) || 0; // EAR formula: average vertical / horizontal
    };

    const leftEyeAR = getEyeAR(landmarks.getLeftEye()); // Left eye EAR
    const rightEyeAR = getEyeAR(landmarks.getRightEye()); // Right eye EAR

    return (leftEyeAR + rightEyeAR) / 2.0; // Average EAR of both eyes
  }

  // Calculates Euclidean distance between two points
  private euclideanDistance(point1: faceapi.Point, point2: faceapi.Point): number {
    return Math.sqrt(Math.pow(point1.x - point2.x, 2) + Math.pow(point1.y - point2.y, 2)); // Distance formula
  }

  // Calculates overall motion between two sets of landmarks
  private calculateMotion(currentLandmarks: faceapi.FaceLandmarks68, previousLandmarks: faceapi.FaceLandmarks68): number {
    const currentPositions = currentLandmarks.positions;
    const previousPositions = previousLandmarks.positions;

    let totalMotion = 0;
    // Sum the motion of each landmark
    for (let i = 0; i < currentPositions.length; i++) {
      const dx = currentPositions[i].x - previousPositions[i].x;
      const dy = currentPositions[i].y - previousPositions[i].y;
      totalMotion += Math.sqrt(dx * dx + dy * dy); // Euclidean distance per landmark
    }
    return totalMotion / currentPositions.length; // Average motion across all landmarks
  }

  // Updates blink detection logic based on EAR and nose motion
  private updateBlinkDetection(eyeAR: number, noseMotion: number): void {
    const currentTime = Date.now();
    const secondsSinceStart = Math.floor((currentTime - this.startTime) / 1000);

    // Check if head is stable (no significant movement)
    let isHeadStable = true;
    if (this.detectionState.lastLandmarks) {
      const currentPosition = this.calculateHeadPosition(this.detectionState.lastLandmarks);
      const motionX = Math.abs(currentPosition.x - this.baselineHeadPosition.x);
      const motionY = Math.abs(currentPosition.y - this.baselineHeadPosition.y);
      isHeadStable = motionX < this.HEAD_MOVEMENT_THRESHOLD && motionY < this.HEAD_MOVEMENT_THRESHOLD;
    }

    console.debug(`EAR: ${eyeAR.toFixed(3)}, Nose Motion: ${noseMotion.toFixed(2)}, Head Stable: ${isHeadStable}`);

    // Detect a blink if EAR is below threshold, head is stable, and nose motion is minimal
    if (eyeAR < this.BLINK_SETTINGS.EYE_ASPECT_RATIO_THRESHOLD && isHeadStable && noseMotion < this.NOSE_MOTION_THRESHOLD) {
      this.consecutiveBlinkFrames++; // Increment blink frame counter

      // Confirm blink if conditions are met
      if (
        this.consecutiveBlinkFrames >= this.BLINK_SETTINGS.MIN_BLINK_FRAMES &&
        this.consecutiveBlinkFrames <= this.BLINK_SETTINGS.MAX_BLINK_FRAMES &&
        !this.detectionState.isBlinking &&
        currentTime - this.detectionState.lastBlinkTime > this.BLINK_SETTINGS.MIN_BLINK_INTERVAL
      ) {
        this.detectionState.isBlinking = true; // Mark as blinking
        this.detectionState.lastBlinkTime = currentTime; // Update last blink time
        this.detectionState.blinkCount++; // Increment blink count
        console.debug(`Blink detected! EAR: ${eyeAR.toFixed(3)}, Count: ${this.detectionState.blinkCount}`);

        // Capture image slightly delayed to get closed eyes
        setTimeout(() => this.captureImage('blink'), 200);

        // Record blink in results if within test duration
        if (secondsSinceStart < this.TOTAL_TEST_DURATION / 1000) {
          this.results[secondsSinceStart + 1] = 'yes';
        }
        this.cdr.detectChanges();
      }
    } else {
      // Reset blink detection if frames exceed minimum or conditions fail
      if (this.consecutiveBlinkFrames >= this.BLINK_SETTINGS.MIN_BLINK_FRAMES) {
        this.consecutiveBlinkFrames = 0;
        this.detectionState.isBlinking = false;
        console.debug(`Blink reset. EAR: ${eyeAR.toFixed(3)}, Head Stable: ${isHeadStable}`);
        this.cdr.detectChanges();
      }
    }

    this.detectionState.currentEyeDistance = eyeAR; // Update current EAR
  }

  // Draws detection results and overlays on the canvas
  private drawResults(landmarks: faceapi.FaceLandmarks68): void {
    if (!this.canvasElement?.nativeElement) return; // Exit if canvas is missing
    const canvas = this.canvasElement.nativeElement;
    const ctx = canvas.getContext('2d');
    if (!ctx) return; // Exit if context is unavailable

    ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear previous drawings

    // Draw eye outlines and points during detection (not calibration)
    if (!this.isCalibrating) {
      ctx.fillStyle = '#00FF00'; // Green color
      const pointRadius = 2;

      const drawEye = (eye: faceapi.Point[]) => {
        ctx.beginPath();
        ctx.moveTo(eye[0].x, eye[0].y);
        for (let i = 1; i < eye.length; i++) {
          ctx.lineTo(eye[i].x, eye[i].y); // Draw outline
        }
        ctx.closePath();
        ctx.strokeStyle = '#00FF00';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Draw points on eye landmarks
        eye.forEach((point) => {
          ctx.beginPath();
          ctx.arc(point.x, point.y, pointRadius, 0, 2 * Math.PI);
          ctx.fill();
        });
      };

      drawEye(landmarks.getLeftEye());
      drawEye(landmarks.getRightEye());
    }

    // Draw direction arrow during head movement stage
    if (this.detectionState.stage === 'headMovement') {
      this.drawDirectionArrow(ctx, canvas.width, canvas.height);
    }

    // Set text styling
    ctx.fillStyle = '#FFFFFF'; // White text
    ctx.font = 'bold 16px Arial';
    ctx.strokeStyle = '#000000'; // Black outline
    ctx.lineWidth = 2;

    const drawText = (text: string, x: number, y: number) => {
      ctx.strokeText(text, x, y); // Outline
      ctx.fillText(text, x, y);   // Fill
    };

    // Draw basic status info
    drawText(`Stage: ${this.detectionState.stage}`, 10, 30);
    drawText(`Time Remaining: ${this.timeRemaining}s`, 10, 50);

    // Draw distance/lighting message in red if present
    if (this.distanceMessage) {
      ctx.fillStyle = '#FF0000';
      drawText(this.distanceMessage, 10, canvas.height - 40);
      ctx.fillStyle = '#FFFFFF'; // Reset color
    }

    // Draw blinking stage info
    if (this.detectionState.stage === 'blinking') {
      drawText(`Blinks Detected: ${this.detectionState.blinkCount}`, 10, 70);
      drawText(`Eye Ratio: ${this.detectionState.currentEyeDistance.toFixed(3)}`, 10, 90);

      if (this.detectionState.isBlinking) {
        ctx.fillStyle = '#FF0000';
        drawText('BLINK DETECTED!', canvas.width / 2 - 60, 30); // Highlight active blink
      }
    }

    // Draw head movement stage info
    if (this.detectionState.stage === 'headMovement') {
      drawText(`Required Movements: ${this.detectionState.requiredMovements.join(', ')}`, 10, 70);
      drawText(`Completed: ${this.detectionState.completedMovements.join(', ')}`, 10, 90);

      if (this.detectionState.currentMovementDirection) {
        ctx.fillStyle = '#FFFF00'; // Yellow text
        drawText(`Current Movement: ${this.detectionState.currentMovementDirection}`, 10, 110);
      }
    }

    // Draw calibration overlay
    if (this.isCalibrating) {
      const positions = landmarks.positions;
      const minX = Math.min(...positions.map((p) => p.x)); // Bounding box min X
      const maxX = Math.max(...positions.map((p) => p.x)); // Bounding box max X
      const minY = Math.min(...positions.map((p) => p.y)); // Bounding box min Y
      const maxY = Math.max(...positions.map((p) => p.y)); // Bounding box max Y
      const padding = 20;
      const boxX = minX - padding;
      const boxY = minY - padding;
      const boxWidth = maxX - minX + 2 * padding;
      const boxHeight = maxY - minY + 2 * padding;

      // Draw face bounding box
      ctx.strokeStyle = '#00BFFF'; // Blue
      ctx.lineWidth = 2;
      ctx.strokeRect(boxX, boxY, boxWidth, boxHeight);

      // Draw scanning grid
      ctx.strokeStyle = 'rgba(0, 191, 255, 0.3)';
      for (let y = boxY; y <= boxY + boxHeight; y += 20) {
        ctx.beginPath();
        ctx.moveTo(boxX, y);
        ctx.lineTo(boxX + boxWidth, y);
        ctx.stroke();
      }
      for (let x = boxX; x <= boxX + boxWidth; x += 20) {
        ctx.beginPath();
        ctx.moveTo(x, boxY);
        ctx.lineTo(x, boxY + boxHeight);
        ctx.stroke();
      }

      // Draw animated scanning line
      const time = Date.now();
      const scanningLineOffset = (Math.sin(time * this.scanningSpeed) + 1) / 2; // Oscillates 0-1
      const scanningLineY = boxY + scanningLineOffset * boxHeight;

      ctx.beginPath();
      ctx.moveTo(boxX, scanningLineY);
      ctx.lineTo(boxX + boxWidth, scanningLineY);
      ctx.strokeStyle = '#00BFFF';
      ctx.lineWidth = 4;
      ctx.stroke();

      // Draw calibration progress
      drawText(`Scanning Face: ${this.calibrationProgress.toFixed(0)}%`, 10, canvas.height - 20);
    }
  }

  // Draws an arrow indicating the current head movement direction
  private drawDirectionArrow(ctx: CanvasRenderingContext2D, width: number, height: number): void {
    if (!this.detectionState.currentMovementDirection) return; // Exit if no direction

    const centerX = width / 2; // Canvas center X
    const centerY = height / 2; // Canvas center Y
    const arrowSize = 50; // Length of arrow
    const arrowWidth = 15; // Width of arrowhead

    ctx.save(); // Save context state
    ctx.strokeStyle = '#FFFF00'; // Yellow
    ctx.fillStyle = '#FFFF00';
    ctx.lineWidth = 3;

    const drawArrow = (fromX: number, fromY: number, toX: number, toY: number) => {
      ctx.beginPath();
      ctx.moveTo(fromX, fromY);
      ctx.lineTo(toX, toY);
      ctx.stroke(); // Draw arrow shaft

      // Calculate arrowhead
      const angle = Math.atan2(toY - fromY, toX - fromX);
      ctx.beginPath();
      ctx.moveTo(toX, toY);
      ctx.lineTo(toX - arrowWidth * Math.cos(angle - Math.PI / 6), toY - arrowWidth * Math.sin(angle - Math.PI / 6));
      ctx.lineTo(toX - arrowWidth * Math.cos(angle + Math.PI / 6), toY - arrowWidth * Math.sin(angle + Math.PI / 6));
      ctx.closePath();
      ctx.fill(); // Fill arrowhead
    };

    // Draw arrow based on direction
    switch (this.detectionState.currentMovementDirection) {
      case 'up':
        drawArrow(centerX, centerY + arrowSize / 2, centerX, centerY - arrowSize / 2);
        break;
      case 'down':
        drawArrow(centerX, centerY - arrowSize / 2, centerX, centerY + arrowSize / 2);
        break;
      case 'left':
        drawArrow(centerX - arrowSize / 2, centerY, centerX + arrowSize / 2, centerY);
        break;
      case 'right':
        drawArrow(centerX + arrowSize / 2, centerY, centerX - arrowSize / 2, centerY);
        break;
    }

    ctx.restore(); // Restore context state
  }

  // Resets the component state to initial values
  private resetState(): void {
    this.detectionState = {
      isBlinking: false,
      lastBlinkTime: 0,
      blinkCount: 0,
      currentEyeDistance: 0,
      baselineEyeDistance: this.storedCalibration ? this.storedCalibration.baselineEyeDistance : 0, // Retain baseline if available
      calibrationFrames: [],
      motionDetected: false,
      stage: 'Scanning',
      requiredMovements: [],
      completedMovements: [],
      currentMovementDirection: null,
    };

    this.results = {}; // Clear results
    this.finalResult = null; // Clear final result
    this.error = null; // Clear error
    this.consecutiveBlinkFrames = 0; // Reset blink frame counter
    this.frameCount = 0; // Reset frame counter
    this.eyeDistanceWindow.length = 0; // Clear EAR window
    this.staticFrameCounter = 0; // Reset static frame counter
    this.motionBuffer = []; // Clear motion buffer
    this.motionCount = 0; // Reset motion count
    this.baselineHeadPosition = { x: 0, y: 0, z: 0 }; // Reset head position
    this.timeRemaining = 0; // Reset time remaining
    this.capturedImages = []; // Clear captured images
    this.selectedImage = null; // Clear selected image
    this.matchDetails = null; // Clear match details
    this.showRetryButton = false; // Hide retry button
    this.showGoAheadButton = false; // Hide go-ahead button
    this.recordedVideoBase64 = null; // Clear recorded video

    // Clear stage timer if active
    if (this.stageTimer) {
      clearTimeout(this.stageTimer);
      this.stageTimer = null;
    }
    this.cdr.detectChanges(); // Update UI
  }

  // Completes the test and evaluates results
  private completeTest(): void {
    this.detectionActive = false; // Stop detection
    this.detectionState.stage = 'complete'; // Mark as complete
    const allMovementsCompleted = this.detectionState.completedMovements.length === 2; // Check if both movements done
    const totalBlinks = this.detectionState.blinkCount; // Total blinks detected

    // Stop video recording if active
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.stop();
    }

    // Evaluate success
    if (allMovementsCompleted && totalBlinks > 0) {
      // Get blink match percentages
      const blinkPercentages = this.capturedImages
        .filter(img => img.event === 'blink' && img.matchPercentage)
        .map(img => img.matchPercentage!.toFixed(2));

      // Initial success message
      this.finalResult = `Liveness verification successful!\n` +
        `- Total Blinks: ${totalBlinks}\n` +
        `- Blink Match Percentages: [${blinkPercentages.join(', ')}]\n` +
        `- Completed Head Movements: ${this.detectionState.completedMovements.join(', ')}\n` +
        `- Attempts: ${this.attemptCount}/${this.MAX_ATTEMPTS}`;

      // Prepare data for backend
      const imagesToSend = this.capturedImages.map((img) => ({
        base64: img.base64,
        timestamp: img.timestamp,
        event: img.event,
        movement: img.movement,
      }));

      const payload = {
        images: imagesToSend,
        video: this.recordedVideoBase64 || '', // Include video if available
      };

      // Send data to backend for verification
      this.http.post<BackendResponse>('http://localhost:3000/upload-images', payload).subscribe({
        next: (response: BackendResponse) => {
          this.matchDetails = response; // Store backend response
          // Update captured images with match percentages
          this.capturedImages.forEach((img, index) => {
            if (img.event === 'blink') {
              const matchData = response.details.capturedData.find(
                (data, dataIndex) => {
                  if (data.timestamp) {
                    return data.event === img.event && data.timestamp === img.timestamp;
                  }
                  return data.event === img.event && dataIndex === index; // Fallback to index if no timestamp
                }
              );
              if (matchData) img.matchPercentage = matchData.matchPercentage;
            }
          });

          // Update blink percentages with backend data
          const updatedBlinkPercentages = this.capturedImages
            .filter(img => img.event === 'blink' && img.matchPercentage)
            .map(img => img.matchPercentage!.toFixed(2));

          // Update final result with backend data
          this.finalResult = `Liveness verification successful!\n` +
            `- Total Blinks: ${totalBlinks}\n` +
            `- Blink Match Percentages: [${updatedBlinkPercentages.join(', ')}]\n` +
            `- Completed Head Movements: ${this.detectionState.completedMovements.join(', ')}\n` +
            `- Attempts: ${this.attemptCount}/${this.MAX_ATTEMPTS}` +
            `\n- Face Match: ${response.match ? 'Yes' : 'No'}` +
            `\n- Average Match Percentage: ${response.averageMatchPercentage.toFixed(2)}%`;

          // Determine next steps based on match percentage
          if (response.averageMatchPercentage >= this.MATCH_THRESHOLD) {
            this.showGoAheadButton = true; // Show go-ahead button
            this.showRetryButton = false;
          } else {
            this.showRetryButton = true; // Show retry button
            this.showGoAheadButton = false;
            this.finalResult += `\n- Face match below ${this.MATCH_THRESHOLD}% threshold. Please try again.`;
          }

          // Handle max attempts
          if (this.attemptCount >= this.MAX_ATTEMPTS) {
            this.maxAttemptsReached = true;
            this.showRetryButton = false;
            this.showGoAheadButton = false;
            this.finalResult = `Maximum attempts (${this.MAX_ATTEMPTS}) reached.\n` +
              `- Face match failed with ${response.averageMatchPercentage.toFixed(2)}%\n` +
              `- Please contact support`;
          }
          this.cdr.detectChanges();
        },
        error: (error: HttpErrorResponse) => {
          // Handle backend communication error
          this.finalResult += `\n- Error: Failed to send data to server. ${error.status} - ${error.message || 'Unknown error'}`;
          this.showRetryButton = true;
          this.showGoAheadButton = false;
          this.cdr.detectChanges();
        },
      });
    } else {
      // Handle incomplete test cases
      if (totalBlinks === 0) {
        this.finalResult = `Test incomplete. No blinks detected during the test. Attempts: ${this.attemptCount}/${this.MAX_ATTEMPTS}`;
      } else if (!allMovementsCompleted) {
        this.finalResult = `Test incomplete. Not all required head movements were completed. Attempts: ${this.attemptCount}/${this.MAX_ATTEMPTS}`;
      } else {
        this.finalResult = `Test incomplete. Please ensure you complete both blinks and head movements. Attempts: ${this.attemptCount}/${this.MAX_ATTEMPTS}`;
      }
      this.showRetryButton = true;
      this.showGoAheadButton = false;
      this.cdr.detectChanges();
    }

    // Clear stage timer
    if (this.stageTimer) {
      clearTimeout(this.stageTimer);
      this.stageTimer = null;
    }
  }

  // Handles errors by stopping processes and displaying a message
  private handleError(message: string): void {
    this.error = message; // Set error message
    this.isLoading = false; // Stop loading state
    this.detectionActive = false; // Stop detection
    console.error(message);

    // Stop video recording if active
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.stop();
    }

    // Clear stage timer
    if (this.stageTimer) {
      clearTimeout(this.stageTimer);
      this.stageTimer = null;
    }

    this.stopCameraStream(); // Stop camera
    this.cdr.detectChanges(); // Update UI
  }

  // Retries the detection process
  public retry(): void {
    console.debug('Retrying detection...');
    this.stopCameraStream(); // Stop current stream
    this.resetState(); // Reset all state
    this.error = null; // Clear error
    this.isLoading = false; // Reset loading state
    this.maxAttemptsReached = false; // Reset max attempts flag
    this.cdr.detectChanges();

    // Reinitialize detector asynchronously
    setTimeout(async () => {
      if (!this.videoElement || !this.videoElement.nativeElement || !this.canvasElement || !this.canvasElement.nativeElement) {
        this.handleError('Video or canvas element not found during retry.');
        return;
      }
      try {
        await this.initializeDetector(); // Restart detector
      } catch (error) {
        this.handleError(`Retry failed: ${(error as Error).message}`);
      }
    }, 0);
  }

  // Proceeds after successful verification
  public goAhead(): void {
    this.stopCameraStream(); // Stop camera
    window.close(); // Close the window (assuming this is a popup)
    this.finalResult = 'Face match successful! Proceeding...';
    this.showGoAheadButton = false; // Hide go-ahead button
    this.showRetryButton = false; // Hide retry button
    this.cdr.detectChanges();
  }

  // Lifecycle hook: Cleans up resources when component is destroyed
  ngOnDestroy(): void {
    this.detectionActive = false; // Stop detection
    this.destroy$.next(); // Signal destruction
    this.destroy$.complete(); // Complete subject

    // Cancel animation frame if active
    if (this.animationFrameId) cancelAnimationFrame(this.animationFrameId);
    // Clear stage timer
    if (this.stageTimer) clearTimeout(this.stageTimer);
    // Stop video recording
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') this.mediaRecorder.stop();

    this.stopCameraStream(); // Stop camera
    this.storedCalibration = null; // Clear calibration data
    console.debug('Calibration data cleared on destroy.');
  }
}
