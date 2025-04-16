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
let runningMode = "VIDEO";
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
async function createImageSegmenter() {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
  );

  imageSegmenter = await ImageSegmenter.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
      delegate: "GPU",
    },
    runningMode,
    outputCategoryMask: true,
    outputConfidenceMasks: false,
  });

  console.log("Model labels:", imageSegmenter.getLabels());
}

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

const HAIR_LABEL = 1; // Set this to match your model's hair label

function callbackForVideo(result) {
  // Log the result to confirm the structure
  console.log(result);

  // Ensure categoryMask exists and is properly structured
  if (!result?.categoryMask) {
    console.error("Error: categoryMask not available", result);
    return; // Exit if the categoryMask isn't available
  }

  const mask = result.categoryMask.getAsFloat32Array();
  
  // Check if mask is valid
  if (!mask || mask.length === 0) {
    console.error("Error: Invalid or empty categoryMask");
    return; // Exit if mask is invalid or empty
  }

  // Get image data from the canvas
  const imageData = canvasCtx.getImageData(
    0,
    0,
    video.videoWidth,
    video.videoHeight
  );
  const data = imageData.data;

  for (let i = 0, j = 0; i < mask.length; i++, j += 4) {
    // Map mask value to category (e.g., hair = 1)
    const maskVal = Math.round(mask[i] * 255.0); // Convert mask value to label

    // Only process hair (assuming 'hair' is the label at index 1)
    if (maskVal === 1 && selectedColor) {  // Check for hair label
      const x = i % video.videoWidth;
      const y = Math.floor(i / video.videoWidth);

      const originalColor = [
        data[j],     // R
        data[j + 1], // G
        data[j + 2], // B
        data[j + 3], // A
      ];

      // Blend with selected hair color
      data[j] = blend(originalColor[0], selectedColor[0], 0.5);
      data[j + 1] = blend(originalColor[1], selectedColor[1], 0.5);
      data[j + 2] = blend(originalColor[2], selectedColor[2], 0.5);

      // Apply hair texture and lighting
      data[j] = addHairTexture(data[j]);
      data[j + 1] = addHairTexture(data[j + 1]);
      data[j + 2] = addHairTexture(data[j + 2]);

      // Apply exposure lighting
      data[j] = applyExposureLighting(data[j], i, video.videoWidth, video.videoHeight);
      data[j + 1] = applyExposureLighting(data[j + 1], i, video.videoWidth, video.videoHeight);
      data[j + 2] = applyExposureLighting(data[j + 2], i, video.videoWidth, video.videoHeight);

      // Edge smoothing
      data[j] = smoothEdges(data[j], originalColor[0]);
      data[j + 1] = smoothEdges(data[j + 1], originalColor[1]);
      data[j + 2] = smoothEdges(data[j + 2], originalColor[2]);

      // Optional: Draw hair streaks
    }
  }

  canvasCtx.putImageData(imageData, 0, 0);

  if (webcamRunning) {
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
