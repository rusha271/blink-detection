// src/image-capture.d.ts
declare class ImageCapture {
    constructor(videoTrack: MediaStreamTrack);
    grabFrame(): Promise<ImageBitmap>;
  }
  
  interface Window {
    ImageCapture?: typeof ImageCapture;
  }