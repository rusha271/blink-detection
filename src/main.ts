import { bootstrapApplication } from '@angular/platform-browser';
import { provideHttpClient } from '@angular/common/http';
import { BlinkDetectorComponent } from './app/blink-detector/blink-detector.component';

bootstrapApplication(BlinkDetectorComponent, {
  providers: [
    provideHttpClient()
  ]
}).catch(err => console.error(err));