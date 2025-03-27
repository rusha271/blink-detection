declare module '@tensorflow-models/facemesh' {
    import * as tf from '@tensorflow/tfjs';

    export interface Prediction {
        faceInViewConfidence: number;
        boundingBox: {
            topLeft: [number, number];
            bottomRight: [number, number];
        };
        mesh: [number, number, number][];
        scaledMesh: [number, number, number][];
    }

    export class FaceMesh {
        estimateFaces(
            input: tf.Tensor3D | ImageData | HTMLVideoElement | HTMLImageElement | HTMLCanvasElement,
            config?: { flipHorizontal?: boolean; predictIrises?: boolean }
        ): Promise<Prediction[]>;
    }

    export function load(config?: { maxFaces?: number }): Promise<FaceMesh>;
}
