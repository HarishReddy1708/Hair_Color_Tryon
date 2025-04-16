import {
  ImageSegmenter,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2";
//import cv from 'https://docs.opencv.org/master/opencv.js';
//import { cv } from 'https://docs.opencv.org/master/opencv.js';

function cvLoaded() {
  console.log("OpenCV.js is loaded!");
}
// DOM elements
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("canvas");
canvasElement.willReadFrequently = true;
const canvasCtx = canvasElement.getContext("2d");
const webcamPredictions = document.getElementById("webcamPredictions");
const enableWebcamButton = document.getElementById("webcamButton");

let webcamRunning = false;
let runningMode = "IMAGE";
const resultWidthHeight = 256;
let imageSegmenter;
let labels;

// Default color to apply
let selectedColor = null;
let prevmask = null;

const legendColors = [
  [30, 0, 7, 255], //Black
  [90, 56, 37, 255], // light Brown
  [62, 0, 0, 255], // Dark Brown
  [110, 0, 0, 255], // Dark Burgundy
  [124, 0, 38, 255], // light Burgundy
  [130, 0, 110, 255], // magenta
  [75, 0, 117, 255], // Voilet
  [20, 10, 90, 255], //Blue
];

// Load the image segmenter
const createImageSegmenter = async () => {
  try {
    // Use the local path for serving the WASM files
    const audio = await FilesetResolver.forVisionTasks("/tasks-vision/wasm");

    imageSegmenter = await ImageSegmenter.createFromOptions(audio, {
      baseOptions: {
        modelAssetPath: "/model/hair_segmenter.tflite",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      outputCategoryMask: true,
      outputConfidenceMasks: true,
      hairOnly: true,
    });
    labels = imageSegmenter.getLabels();
    //demosSection.classList.remove("invisible");
  } catch (error) {
    console.error("Error loading MediaPipe Image Segmenter:", error);
  }
};

// Webcam segmentation
let lastUpdateTime = 0;
const updateInterval = 20;
let frmcnt = 1;
async function predictWebcam() {
  //await loadOpenCV();
  const currentTime = performance.now();
  if (currentTime - lastUpdateTime < updateInterval) {
    if (webcamRunning) {
      window.requestAnimationFrame(predictWebcam);
    }
    return;
  }
  lastUpdateTime = currentTime;

  // Continue with the segmentation logic...
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

  if (imageSegmenter === undefined) return;

  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    imageSegmenter.setOptions({
      runningMode: runningMode,
    });
  }

  let startTimeMs = performance.now();
  imageSegmenter.segmentForVideo(video, startTimeMs, callbackForVideo);
}

function blend(original, selected, alpha) {
  return Math.round(original * (1 - alpha) + selected * alpha);
}

function callbackForVideo(result) {
  if (!result?.confidenceMasks[1]) return;
  let mask = null;
  if (frmcnt % 3 == 0 || frmcnt == 1 || prevmask == null) {
    mask = result.confidenceMasks[1].getAsFloat32Array();
    prevmask = mask;
  } else mask = prevmask;

  let imageData = canvasCtx.getImageData(
    0,
    0,
    video.videoWidth,
    video.videoHeight
  ).data;
  let data = imageData.data;
  let threshold = 0.5;
  for (let i = 0, j = 0; i < mask.length; i++, j += 4) {
    if (mask[i] >= threshold && selectedColor) {
      const originalColor = [
        imageData[j], // R
        imageData[j + 1], // G
        imageData[j + 2], // B
        imageData[j + 3], // A
      ];
      let alpha = 0.6;
      if (mask[i] >= 0.8) alpha = 0.6;
      else if (mask[i] > 0.7) alpha = 0.55;
      else alpha = 0.5;
      imageData[j] = blend(imageData[j], selectedColor[0], alpha);
      imageData[j + 1] = blend(imageData[j + 1], selectedColor[1], alpha);
      imageData[j + 2] = blend(imageData[j + 2], selectedColor[2], alpha);

      imageData[j] = addHairTexture(imageData[j]);
      imageData[j + 1] = addHairTexture(imageData[j + 1]);
      imageData[j + 2] = addHairTexture(imageData[j + 2]);

      imageData[j] = applyExposureLighting(
        imageData[j],
        i,
        video.videoWidth,
        video.videoHeight
      );
      imageData[j + 1] = applyExposureLighting(
        imageData[j + 1],
        i,
        video.videoWidth,
        video.videoHeight
      );
      imageData[j + 2] = applyExposureLighting(
        imageData[j + 2],
        i,
        video.videoWidth,
        video.videoHeight
      );

      imageData[j] = smoothEdges(imageData[j], originalColor[0]);
      imageData[j + 1] = smoothEdges(imageData[j + 1], originalColor[1]);
      imageData[j + 2] = smoothEdges(imageData[j + 2], originalColor[2]);
    }
  }

  // Ensure the image data is valid
  const uint8Array = new Uint8ClampedArray(imageData.buffer);
  const dataNew = new ImageData(
    uint8Array,
    video.videoWidth,
    video.videoHeight
  );
  canvasCtx.putImageData(dataNew, 0, 0);

  // Re-request the next frame if webcam is running
  if (webcamRunning === true) {
    window.requestAnimationFrame(predictWebcam);
  }
}
function addHairTexture(color) {
  const noise = Math.random() * 10 - 40;
  return Math.min(255, Math.max(0, color + noise));
}
function applyExposureLighting(color, pixelIndex, width, height) {
  const x = pixelIndex % width;
  const y = Math.floor(pixelIndex / width);
  const lightX = width / 2;
  const lightY = height / 3;
  const maxDistance = Math.sqrt((x - lightX) ** 2 + (y - lightY) ** 2);
  const distanceFactor = 1 - maxDistance / (width / 2);
  const shineFactor = Math.pow(distanceFactor, 2);
  const exposureFactor = Math.pow(distanceFactor, 3);
  return Math.min(255, color * (1 + exposureFactor * 3 + shineFactor * 1.2));
}

function smoothEdges(current, original) {
  const blendRatio = 0.8;
  return Math.round(current * (1 - blendRatio) + original * blendRatio);
}

// Enable the live webcam view
async function enableCam(event) {
  if (imageSegmenter === undefined) {
    return;
  }

  if (webcamRunning === true) {
    webcamRunning = false;
    enableWebcamButton.innerText = "Apply";
  } else {
    webcamRunning = true;
    enableWebcamButton.innerText = "Reset";
  }

  const constraints = {
    video: {
      facingMode: "environment",
    },
  };

  try {
    video.srcObject = await navigator.mediaDevices.getUserMedia(constraints);
    video.addEventListener("loadeddata", predictWebcam);
  } catch (err) {
    console.error("Error accessing webcam: ", err);
    alert(
      "Failed to access webcam. Please ensure your browser has the correct permissions."
    );
  }
}

document.getElementById("colorBlack").addEventListener("click", () => {
  selectedColor = legendColors[0]; // Black
});
document.getElementById("colorLightBrown").addEventListener("click", () => {
  selectedColor = legendColors[1]; // Light Brown
});
document.getElementById("colorDarkBrown").addEventListener("click", () => {
  selectedColor = legendColors[2]; // Dark Brown
});
document.getElementById("colorDarkBurgundy").addEventListener("click", () => {
  selectedColor = legendColors[3]; // Dark Burgundy
});
document.getElementById("colorLightBurgundy").addEventListener("click", () => {
  selectedColor = legendColors[4]; // Light Burgundy
});
document.getElementById("colormagenta").addEventListener("click", () => {
  selectedColor = legendColors[5]; // Magenta
});
document.getElementById("colovoilet").addEventListener("click", () => {
  selectedColor = legendColors[6]; // Violet
});
document.getElementById("colorBlue").addEventListener("click", () => {
  selectedColor = legendColors[7]; // Blue
});

// Start the segmentation process
createImageSegmenter();

// Check if webcam access is supported
function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

if (hasGetUserMedia()) {
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}
