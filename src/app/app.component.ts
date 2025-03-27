import { Component } from '@angular/core';
import { BlinkDetectorComponent } from './blink-detector/blink-detector.component';

@Component({
  selector: 'app-root',
  template: '<app-blink-detector></app-blink-detector>',
  standalone: true,
  imports: [BlinkDetectorComponent]
})
export class AppComponent {}