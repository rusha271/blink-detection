import { ComponentFixture, TestBed } from '@angular/core/testing';

import { BlinkDetectorComponent } from './blink-detector.component';

describe('BlinkDetectorComponent', () => {
  let component: BlinkDetectorComponent;
  let fixture: ComponentFixture<BlinkDetectorComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [BlinkDetectorComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(BlinkDetectorComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
