/**
 * AI Crop Doctor — Frontend Application Logic
 * Vanilla JavaScript (ES6+) with Live Camera, Audio Waveform Recorder,
 * Speech Synthesis Player, and Confidence-Aware Diagnostic Visualizer.
 *
 * Supports Dynamic Modality Matching:
 * - If input was provided via Voice/Mic -> Response is spoken aloud via Speech (TTS)
 * - If input was provided via Text -> Response is given in Text (no auto-speech)
 */

// Dynamic API Base URL (works whether loaded on FastAPI :8000, Live Server :5500, or file://)
const API_BASE = (window.location.protocol.startsWith("http") && (window.location.port === "8000" || (window.location.port === "" && window.location.hostname !== "")))
  ? ""
  : "http://127.0.0.1:8000";

// Application State
const AppState = {
  caseId: null,
  selectedImageFile: null,
  cameraStream: null,
  mediaRecorder: null,
  audioChunks: [],
  isRecording: false,
  audioBlob: null,
  audioContext: null,
  analyser: null,
  animationFrameId: null,
  lastInputWasVoice: false,
  currentDiagnosis: null,
};

// Preset Sample Cases for 1-Click Instant Testing
const SamplePresets = {
  tomato_early_blight: {
    name: "Tomato (Early Blight)",
    symptoms: "Older lower leaves have dark brown concentric spots like target rings. Leaves turning yellow and dropping.",
    plant: "Tomato",
    svgColor: "#84cc16",
    leafType: "concentric"
  },
  potato_late_blight: {
    name: "Potato (Late Blight)",
    symptoms: "Water-soaked dark lesions on leaf tips and margins with white fuzzy mold underneath during damp mornings.",
    plant: "Potato",
    svgColor: "#10b981",
    leafType: "water_soaked"
  },
  corn_common_rust: {
    name: "Corn / Maize (Common Rust)",
    symptoms: "Small, circular to elongated cinnamon-brown pustules scattered over both upper and lower leaf surfaces.",
    plant: "Corn",
    svgColor: "#f59e0b",
    leafType: "pustules"
  },
  apple_scab: {
    name: "Apple (Apple Scab)",
    symptoms: "Velvety olive-green to dark brown spots with irregular margins on leaves, causing premature yellowing.",
    plant: "Apple",
    svgColor: "#22c55e",
    leafType: "olive_spots"
  },
  hindi_tomato_blight: {
    name: "🌱 [हिंदी] टमाटर (अगेती झुलसा)",
    symptoms: "टमाटर के निचले पत्तों पर भूरे और काले छल्ले जैसे धब्बे बन रहे हैं और पत्ते पीले होकर गिर रहे हैं।",
    plant: "Tomato / टमाटर",
    svgColor: "#ef4444",
    leafType: "concentric"
  },
  hindi_potato_blight: {
    name: "🌱 [हिंदी] आलू (पछेती झुलसा)",
    symptoms: "आलू के पत्तों पर पानी से भीगे हुए काले धब्बे हैं और सुबह के समय पत्तों के नीचे सफेद ففूंद दिखती है।",
    plant: "Potato / आलू",
    svgColor: "#059669",
    leafType: "water_soaked"
  },
  punjabi_tomato_blight: {
    name: "🌾 [ਪੰਜਾਬੀ] ਟਮਾਟਰ (ਅਗੇਤੀ ਝੁਲਸਾ)",
    symptoms: "ਟਮਾਟਰ ਦੇ ਹੇਠਲੇ ਪੱਤਿਆਂ ਉੱਤੇ ਭੂਰੇ ਅਤੇ ਕਾਲੇ ਛੱਲਿਆਂ ਵਰਗੇ ਧੱਬੇ ਬਣ ਰਹੇ ਹਨ ਅਤੇ ਪੱਤੇ ਪੀਲੇ ਹੋ ਕੇ ਡਿੱਗ ਰਹੇ ਹਨ।",
    plant: "Tomato / ਟਮਾਟਰ",
    svgColor: "#dc2626",
    leafType: "concentric"
  },
  punjabi_potato_blight: {
    name: "🌾 [ਪੰਜਾਬੀ] ਆਲੂ (ਪਿਛੇਤੀ ਝੁਲਸਾ)",
    symptoms: "ਆਲੂ ਦੇ ਪੱਤਿਆਂ ਉੱਤੇ ਪਾਣੀ ਨਾਲ ਭਿੱਜੇ ਹੋਏ ਕਾਲੇ ਧੱਬੇ ਹਨ ਅਤੇ ਸਵੇਰੇ ਪੱਤਿਆਂ ਦੇ ਹੇਠਾਂ ਚਿੱਟੀ ਉੱਲੀ ਦਿਖਦੀ ਹੈ।",
    plant: "Potato / ਆਲੂ",
    svgColor: "#047857",
    leafType: "water_soaked"
  }
};

// DOM Elements
const DOM = {
  // Brand & Header
  healthStatusPill: document.getElementById("health-status-pill"),
  statusText: document.getElementById("status-text"),
  statusDot: document.getElementById("status-dot"),
  btnNewCase: document.getElementById("btn-new-case"),
  btnAutoTts: document.getElementById("btn-auto-tts"),

  // Visual Inputs
  tabUpload: document.getElementById("tab-upload"),
  tabCamera: document.getElementById("tab-camera"),
  dropzone: document.getElementById("dropzone"),
  fileInput: document.getElementById("file-input"),
  cameraContainer: document.getElementById("camera-container"),
  cameraVideo: document.getElementById("camera-video"),
  btnSnap: document.getElementById("btn-snap"),
  imagePreviewWrapper: document.getElementById("image-preview-wrapper"),
  previewImg: document.getElementById("preview-img"),
  previewFileName: document.getElementById("preview-file-name"),
  btnClearImage: document.getElementById("btn-clear-image"),

  // Voice Inputs
  btnMic: document.getElementById("btn-mic"),
  audioCanvas: document.getElementById("audio-canvas"),
  audioStatusLabel: document.getElementById("audio-status-label"),

  // Text Inputs & Submit
  symptomInput: document.getElementById("symptom-input"),
  btnSubmit: document.getElementById("btn-submit"),
  submitSpinner: document.getElementById("submit-spinner"),
  submitBtnText: document.getElementById("submit-btn-text"),
  presetChipsContainer: document.getElementById("preset-chips-container"),

  // Timeline / Feed
  chatMessages: document.getElementById("chat-messages"),

  // Prescription Card
  prescriptionPanel: document.getElementById("prescription-panel"),
  rxPlantName: document.getElementById("rx-plant-name"),
  rxDiseaseName: document.getElementById("rx-disease-name"),
  rxSeverityTag: document.getElementById("rx-severity-tag"),
  rxConfidenceProgress: document.getElementById("rx-confidence-progress"),
  rxConfidenceText: document.getElementById("rx-confidence-text"),
  rxConfidenceStatus: document.getElementById("rx-confidence-status"),
  rxCauseText: document.getElementById("rx-cause-text"),
  rxSymptomsList: document.getElementById("rx-symptoms-list"),
  rxOrganicList: document.getElementById("rx-organic-list"),
  rxChemicalList: document.getElementById("rx-chemical-list"),
  rxPreventionList: document.getElementById("rx-prevention-list"),
  rxEvidenceList: document.getElementById("rx-evidence-list"),
  btnSpeakRx: document.getElementById("btn-speak-rx"),
  btnPrintRx: document.getElementById("btnPrintRx"),
};

// ==========================================================================
// Initialization & Health Checks
// ==========================================================================

document.addEventListener("DOMContentLoaded", () => {
  initHealthCheck();
  initVisualInputs();
  initVoiceRecorder();
  initChatTimeline();
  initPresets();
  initActionButtons();
  initTreatmentTabs();
});

async function initHealthCheck() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (res.ok) {
      const data = await res.json();
      DOM.statusText.textContent = "AI Doctor Online";
      DOM.statusDot.style.backgroundColor = "var(--emerald-500)";
      DOM.healthStatusPill.title = `FAISS Loaded: ${data.faiss_loaded}, Embeddings: ${data.embedding_model_loaded}, Chunks: ${data.knowledge_base_chunks}`;
    } else {
      DOM.statusText.textContent = "Service Limited";
      DOM.statusDot.style.backgroundColor = "var(--amber-500)";
    }
  } catch (err) {
    DOM.statusText.textContent = "Connecting to Backend...";
    DOM.statusDot.style.backgroundColor = "var(--amber-500)";
  }
}

// ==========================================================================
// Visual Input & Camera Management (Upload or Take Photo)
// ==========================================================================

function initVisualInputs() {
  // Tabs Switch
  DOM.tabUpload.addEventListener("click", () => {
    DOM.tabUpload.classList.add("active");
    DOM.tabCamera.classList.remove("active");
    DOM.dropzone.style.display = "block";
    stopCamera();
    DOM.cameraContainer.classList.remove("active");
  });

  DOM.tabCamera.addEventListener("click", async () => {
    DOM.tabCamera.classList.add("active");
    DOM.tabUpload.classList.remove("active");
    DOM.dropzone.style.display = "none";
    DOM.cameraContainer.classList.add("active");
    await startCamera();
  });

  // Dropzone Events
  DOM.dropzone.addEventListener("click", () => DOM.fileInput.click());
  DOM.fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleImageSelect(e.target.files[0]);
    }
  });

  DOM.dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    DOM.dropzone.classList.add("dragover");
  });

  DOM.dropzone.addEventListener("dragleave", () => {
    DOM.dropzone.classList.remove("dragover");
  });

  DOM.dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    DOM.dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleImageSelect(e.dataTransfer.files[0]);
    }
  });

  // Clear Image
  DOM.btnClearImage.addEventListener("click", () => {
    clearSelectedImage();
  });

  // Snap Photo from Camera
  DOM.btnSnap.addEventListener("click", () => {
    takeCameraSnapshot();
  });
}

async function startCamera(facingMode = "environment") {
  try {
    if (AppState.cameraStream) {
      stopCamera();
    }
    const constraints = {
      video: { facingMode: facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    };
    AppState.cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
    DOM.cameraVideo.srcObject = AppState.cameraStream;
    DOM.cameraVideo.play();
  } catch (err) {
    console.warn("Camera access denied or unavailable:", err);
    alert("Camera access was not granted or is unsupported. Please upload a crop photo instead.");
    DOM.tabUpload.click();
  }
}

function stopCamera() {
  if (AppState.cameraStream) {
    AppState.cameraStream.getTracks().forEach(track => track.stop());
    AppState.cameraStream = null;
  }
}

function takeCameraSnapshot() {
  if (!AppState.cameraStream) return;
  const canvas = document.createElement("canvas");
  canvas.width = DOM.cameraVideo.videoWidth || 640;
  canvas.height = DOM.cameraVideo.videoHeight || 480;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(DOM.cameraVideo, 0, 0, canvas.width, canvas.height);

  canvas.toBlob((blob) => {
    if (blob) {
      const file = new File([blob], `crop_snap_${Date.now()}.jpg`, { type: "image/jpeg" });
      handleImageSelect(file);
      stopCamera();
      DOM.tabUpload.click();
    }
  }, "image/jpeg", 0.92);
}

function handleImageSelect(file) {
  if (!file || !file.type.startsWith("image/")) {
    alert("Please select a valid image file (JPEG, PNG, WEBP).");
    return;
  }
  AppState.selectedImageFile = file;
  DOM.previewFileName.textContent = file.name;

  const reader = new FileReader();
  reader.onload = (e) => {
    DOM.previewImg.src = e.target.result;
    DOM.imagePreviewWrapper.classList.add("active");
    DOM.dropzone.style.display = "none";
  };
  reader.readAsDataURL(file);
}

function clearSelectedImage() {
  AppState.selectedImageFile = null;
  DOM.previewImg.src = "";
  DOM.fileInput.value = "";
  DOM.imagePreviewWrapper.classList.remove("active");
  DOM.dropzone.style.display = "block";
}

// ==========================================================================
// Preset Sample Generation (SVG & Synthetic Plant Leaf Data)
// ==========================================================================

function initPresets() {
  DOM.presetChipsContainer.innerHTML = "";
  Object.keys(SamplePresets).forEach(key => {
    const preset = SamplePresets[key];
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "preset-chip";
    chip.innerHTML = `<span>🌿</span> ${preset.name}`;
    chip.addEventListener("click", () => loadSamplePreset(key));
    DOM.presetChipsContainer.appendChild(chip);
  });
}

function loadSamplePreset(presetKey) {
  const preset = SamplePresets[presetKey];
  if (!preset) return;

  DOM.symptomInput.value = preset.symptoms;
  AppState.lastInputWasVoice = false; // Mark as text input

  // Generate a high-contrast crop lesion canvas file
  const canvas = document.createElement("canvas");
  canvas.width = 600;
  canvas.height = 450;
  const ctx = canvas.getContext("2d");

  // Background leaf blade gradient
  const grad = ctx.createLinearGradient(0, 0, 600, 450);
  grad.addColorStop(0, "#1b4d1b");
  grad.addColorStop(0.5, preset.svgColor);
  grad.addColorStop(1, "#163816");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 600, 450);

  // Leaf veins
  ctx.strokeStyle = "rgba(255, 255, 255, 0.25)";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(300, 450);
  ctx.lineTo(300, 50);
  ctx.stroke();

  for (let y = 100; y < 400; y += 45) {
    ctx.beginPath();
    ctx.moveTo(300, y);
    ctx.lineTo(100, y - 40);
    ctx.moveTo(300, y);
    ctx.lineTo(500, y - 40);
    ctx.stroke();
  }

  // Draw characteristic pathology lesions
  if (preset.leafType === "concentric") {
    // Early Blight Target Rings
    for (let i = 0; i < 4; i++) {
      const cx = 200 + i * 70;
      const cy = 180 + (i % 2) * 80;
      for (let r = 35; r > 5; r -= 7) {
        ctx.fillStyle = r % 14 === 0 ? "#451a03" : "#78350f";
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  } else if (preset.leafType === "pustules") {
    // Rust Pustules
    for (let i = 0; i < 60; i++) {
      const px = 100 + Math.random() * 400;
      const py = 80 + Math.random() * 300;
      ctx.fillStyle = "#b45309";
      ctx.beginPath();
      ctx.ellipse(px, py, 6, 12, Math.PI / 4, 0, Math.PI * 2);
      ctx.fill();
    }
  } else {
    // General necrotic blotches
    for (let i = 0; i < 6; i++) {
      const bx = 150 + Math.random() * 300;
      const by = 100 + Math.random() * 250;
      ctx.fillStyle = "#292524";
      ctx.beginPath();
      ctx.arc(bx, by, 30 + Math.random() * 25, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Plant & Disease Watermark Badge
  ctx.fillStyle = "rgba(0,0,0,0.6)";
  ctx.fillRect(20, 390, 320, 40);
  ctx.fillStyle = "#ffffff";
  ctx.font = "bold 16px Outfit, sans-serif";
  ctx.fillText(`Sample: ${preset.name}`, 30, 415);

  canvas.toBlob((blob) => {
    const sampleFile = new File([blob], `${presetKey}_sample.jpg`, { type: "image/jpeg" });
    handleImageSelect(sampleFile);
  }, "image/jpeg", 0.9);
}

// ==========================================================================
// Voice Recording & Web Audio Visualizer
// ==========================================================================

function initVoiceRecorder() {
  const canvas = DOM.audioCanvas;
  const canvasCtx = canvas.getContext("2d");

  // Draw idle visualizer line
  function drawIdleLine() {
    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
    canvasCtx.strokeStyle = "rgba(255, 255, 255, 0.15)";
    canvasCtx.lineWidth = 2;
    canvasCtx.beginPath();
    canvasCtx.moveTo(0, canvas.height / 2);
    canvasCtx.lineTo(canvas.width, canvas.height / 2);
    canvasCtx.stroke();
  }
  drawIdleLine();

  DOM.btnMic.addEventListener("click", async () => {
    if (!AppState.isRecording) {
      await startVoiceRecording();
    } else {
      await stopVoiceRecording();
    }
  });

  // If user types directly in symptom box, set modality to text
  DOM.symptomInput.addEventListener("input", () => {
    if (DOM.symptomInput.value.trim().length > 0) {
      // User is manually typing
    }
  });
}

async function startVoiceRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    AppState.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = AppState.audioContext.createMediaStreamSource(stream);
    AppState.analyser = AppState.audioContext.createAnalyser();
    AppState.analyser.fftSize = 256;
    source.connect(AppState.analyser);

    AppState.audioChunks = [];
    AppState.mediaRecorder = new MediaRecorder(stream);

    AppState.mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) AppState.audioChunks.push(e.data);
    };

    AppState.mediaRecorder.onstop = async () => {
      AppState.audioBlob = new Blob(AppState.audioChunks, { type: "audio/wav" });
      AppState.lastInputWasVoice = true; // Mark that user gave voice input!
      stream.getTracks().forEach(t => t.stop());
      cancelAnimationFrame(AppState.animationFrameId);
      DOM.audioStatusLabel.textContent = "Transcribing with Groq Whisper...";

      // Auto-transcribe preview
      await autoTranscribeVoice(AppState.audioBlob);
    };

    AppState.mediaRecorder.start();
    AppState.isRecording = true;
    DOM.btnMic.classList.add("recording");
    DOM.audioStatusLabel.textContent = "Listening... (Click mic when done)";
    visualizeAudio();
  } catch (err) {
    console.error("Microphone access failed:", err);
    alert("Microphone access could not be acquired. You can type symptoms in the text box.");
  }
}

async function stopVoiceRecording() {
  if (AppState.mediaRecorder && AppState.isRecording) {
    AppState.mediaRecorder.stop();
    AppState.isRecording = false;
    DOM.btnMic.classList.remove("recording");
  }
}

function visualizeAudio() {
  const canvas = DOM.audioCanvas;
  const canvasCtx = canvas.getContext("2d");
  const bufferLength = AppState.analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  function renderFrame() {
    if (!AppState.isRecording) return;
    AppState.animationFrameId = requestAnimationFrame(renderFrame);

    AppState.analyser.getByteFrequencyData(dataArray);

    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
    const barWidth = (canvas.width / bufferLength) * 2.5;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const barHeight = (dataArray[i] / 255) * (canvas.height - 4);
      const grad = canvasCtx.createLinearGradient(0, canvas.height, 0, 0);
      grad.addColorStop(0, "rgba(16, 185, 129, 0.8)");
      grad.addColorStop(1, "rgba(59, 130, 246, 0.9)");

      canvasCtx.fillStyle = grad;
      canvasCtx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
      x += barWidth + 1;
    }
  }
  renderFrame();
}

async function autoTranscribeVoice(audioBlob) {
  try {
    const formData = new FormData();
    formData.append("audio", audioBlob, "voice_input.wav");

    const res = await fetch(`${API_BASE}/api/voice/transcribe`, {
      method: "POST",
      body: formData
    });

    if (res.ok) {
      const data = await res.json();
      if (data.text) {
        DOM.symptomInput.value = (DOM.symptomInput.value ? DOM.symptomInput.value + " " : "") + data.text;
        DOM.audioStatusLabel.textContent = `Heard: "${data.text.slice(0, 32)}..."`;
      } else {
        DOM.audioStatusLabel.textContent = "Voice message recorded (Click Send)";
      }
    } else {
      DOM.audioStatusLabel.textContent = "Voice message recorded";
    }
  } catch (err) {
    console.warn("STT error:", err);
    DOM.audioStatusLabel.textContent = "Voice message recorded";
  }
}

// ==========================================================================
// Diagnostic Submission & Modality Matching
// ==========================================================================

function initChatTimeline() {
  // Introductory greeting
  addAgentMessage({
    text: "Hello! I am your AI Crop Doctor. Upload or take a crop photo, type your symptoms, or speak into the microphone to diagnose plant diseases.",
    actionStep: "Doctor Ready",
    needsFollowup: false
  });
}

function initActionButtons() {
  DOM.btnSubmit.addEventListener("click", (e) => {
    e.preventDefault();
    submitDiagnosis();
  });
  // Allow pressing Enter (without Shift) to submit diagnosis
  DOM.symptomInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitDiagnosis();
    }
  });

  DOM.btnNewCase.addEventListener("click", () => {
    if (confirm("Start a new crop diagnostic consultation?")) {
      resetCase();
    }
  });

  DOM.btnSpeakRx.addEventListener("click", () => {
    if (AppState.currentDiagnosis) {
      const speechSummary = `Diagnosis: ${AppState.currentDiagnosis.disease} in ${AppState.currentDiagnosis.plant_name}. Confidence is ${Math.round(AppState.currentDiagnosis.confidence * 100)} percent. Severity is ${AppState.currentDiagnosis.severity}. Treatment: ${AppState.currentDiagnosis.organic_treatment?.[0] || 'Prune affected leaves'}.`;
      playSynthesizedAudio(speechSummary);
    }
  });
}

async function submitDiagnosis() {
  const text = DOM.symptomInput.value.trim();
  const image = AppState.selectedImageFile;
  const audio = AppState.audioBlob;
  const wasVoice = AppState.lastInputWasVoice;

  if (!text && !image && !audio) {
    alert("Please provide at least a crop photo, typed symptom description, or voice message.");
    return;
  }

  // Display Farmer's message bubble
  addFarmerMessage({
    text: text || (wasVoice ? "🎙️ (Voice description recorded)" : "📷 (Crop photo attached)"),
    imageUrl: image ? DOM.previewImg.src : null,
    isVoice: wasVoice
  });

  // Clear inputs for next turn
  DOM.symptomInput.value = "";
  const attachedImage = image;
  const attachedAudio = audio;
  const submittedWasVoice = wasVoice;

  clearSelectedImage();
  AppState.audioBlob = null;
  AppState.lastInputWasVoice = false; // Reset for next message
  DOM.audioStatusLabel.textContent = "Mic ready";

  // Lock Submit Button with Spinner
  setLoadingState(true);

  // Send request to FastAPI backend
  try {
    const formData = new FormData();
    if (text) formData.append("text", text);
    if (attachedImage) formData.append("image", attachedImage, attachedImage.name);
    if (attachedAudio) formData.append("audio", attachedAudio, "farmer_audio.wav");
    if (AppState.caseId) formData.append("case_id", AppState.caseId);

    const response = await fetch(`${API_BASE}/api/diagnosis`, {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      throw new Error(`Server returned HTTP ${response.status}`);
    }

    const data = await response.json();
    AppState.caseId = data.case_id || AppState.caseId;

    // Render Agent response bubble
    addAgentMessage({
      text: data.response_text || "I have analyzed your crop symptoms and prepared recommendations.",
      actionStep: data.diagnosis ? "Diagnosis Confirmed" : "Gathering Evidence",
      needsFollowup: data.needs_followup,
      followupQuestion: data.followup_question,
      audioUrl: null
    });

    // ──────────────────────────────────────────────────────────────────────
    // Dynamic Modality Rule:
    // "Reply should be given in text if input is in text else speech"
    // ──────────────────────────────────────────────────────────────────────
    if (submittedWasVoice && data.response_text) {
      // Input was voice -> Speak response aloud
      playSynthesizedAudio(data.response_text);
    }

    // Render Prescription Card if structured diagnosis is returned
    if (data.diagnosis) {
      AppState.currentDiagnosis = data.diagnosis;
      renderPrescriptionCard(data.diagnosis);

      // If farmer spoke via voice, also read key diagnosis aloud
      if (submittedWasVoice) {
        const diagSpeech = `Identified ${data.diagnosis.disease} with ${Math.round(data.diagnosis.confidence * 100)} percent confidence.`;
        setTimeout(() => playSynthesizedAudio(diagSpeech), 2000);
      }
    }

  } catch (err) {
    console.error("Diagnosis request error:", err);
    addAgentMessage({
      text: `Unable to complete analysis: ${err.message}. Please ensure the backend is running on http://localhost:8000.`,
      actionStep: "Error",
      needsFollowup: false
    });
  } finally {
    setLoadingState(false);
  }
}

function setLoadingState(isLoading) {
  DOM.btnSubmit.disabled = isLoading;
  DOM.submitSpinner.style.display = isLoading ? "inline-block" : "none";
  DOM.submitBtnText.textContent = isLoading ? "AI Doctor Reasoning..." : "Send to Crop Doctor";
}

function resetCase() {
  AppState.caseId = null;
  AppState.currentDiagnosis = null;
  AppState.lastInputWasVoice = false;
  clearSelectedImage();
  DOM.symptomInput.value = "";
  DOM.prescriptionPanel.classList.remove("active");
  DOM.chatMessages.innerHTML = "";
  initChatTimeline();
}

// ==========================================================================
// Timeline Message Rendering
// ==========================================================================

function addFarmerMessage({ text, imageUrl, isVoice }) {
  const row = document.createElement("div");
  row.className = "message-row farmer";

  const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  let imageHtml = "";
  if (imageUrl) {
    imageHtml = `<div style="margin-bottom:0.5rem;border-radius:8px;overflow:hidden;max-height:160px;"><img src="${imageUrl}" style="width:100%;object-fit:cover;"></div>`;
  }

  const voiceBadge = isVoice ? `<span style="font-size:0.75rem;background:rgba(255,255,255,0.2);padding:2px 6px;border-radius:4px;margin-right:4px;">🎙️ Voice</span> ` : "";

  row.innerHTML = `
    <div class="avatar-circle farmer-av">👨‍🌾</div>
    <div class="message-bubble">
      ${imageHtml}
      <div>${voiceBadge}${escapeHtml(text)}</div>
      <div class="message-time">${timeStr}</div>
    </div>
  `;

  DOM.chatMessages.appendChild(row);
  scrollToBottom();
}

function addAgentMessage({ text, actionStep, needsFollowup, followupQuestion }) {
  const row = document.createElement("div");
  row.className = "message-row agent";

  const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  let actionPillHtml = "";
  if (actionStep) {
    actionPillHtml = `<div class="agent-step-pill"><span>⚡</span> ${escapeHtml(actionStep)}</div>`;
  }

  let followupHtml = "";
  if (needsFollowup && followupQuestion) {
    followupHtml = `
      <div class="followup-box">
        <div class="followup-box-title"><span>❓</span> Clarifying Follow-Up:</div>
        <div class="followup-text">${escapeHtml(followupQuestion)}</div>
        <div class="quick-reply-chips">
          <button class="quick-reply-chip" onclick="applyQuickReply('Yes, only on lower leaves')">Yes, lower leaves</button>
          <button class="quick-reply-chip" onclick="applyQuickReply('Appeared 3 days ago after rain')">After recent rain</button>
          <button class="quick-reply-chip" onclick="applyQuickReply('Spreading rapidly to nearby plants')">Spreading fast</button>
        </div>
      </div>
    `;
  }

  row.innerHTML = `
    <div class="avatar-circle agent-av">🌾</div>
    <div class="message-bubble">
      ${actionPillHtml}
      <div>${formatMarkdownish(text)}</div>
      ${followupHtml}
      <div class="audio-player-widget">
        <button class="audio-play-btn" onclick="playSynthesizedAudio('${escapeForJs(text)}')">
          <svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
        </button>
        <span class="audio-track-label">Click to listen (TTS Voice)</span>
      </div>
      <div class="message-time">${timeStr}</div>
    </div>
  `;

  DOM.chatMessages.appendChild(row);
  scrollToBottom();
}

window.applyQuickReply = function (text) {
  DOM.symptomInput.value = text;
  AppState.lastInputWasVoice = false;
  submitDiagnosis();
};

function scrollToBottom() {
  DOM.chatMessages.scrollTop = DOM.chatMessages.scrollHeight;
}

// ==========================================================================
// Speech Synthesis (Groq Orpheus / Web Speech Fallback)
// ==========================================================================

async function playSynthesizedAudio(text) {
  if (!text) return;
  try {
    const formData = new FormData();
    formData.append("text", text);

    const res = await fetch(`${API_BASE}/api/voice/synthesize`, {
      method: "POST",
      body: formData
    });

    if (res.ok && res.headers.get("X-TTS-Failed") !== "true") {
      const blob = await res.blob();
      if (blob.size > 100) {
        const audioUrl = URL.createObjectURL(blob);
        const audio = new Audio(audioUrl);
        audio.play();
        return;
      }
    }
  } catch (e) {
    console.warn("Groq TTS synthesis error, using browser speech synthesis:", e);
  }

  // Browser Web Speech API fallback
  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text.replace(/[*#]/g, ''));
    utterance.rate = 1.0;
    window.speechSynthesis.speak(utterance);
  }
}

// ==========================================================================
// Confidence-Aware Prescription Card Rendering
// ==========================================================================

function renderPrescriptionCard(dx) {
  DOM.prescriptionPanel.classList.add("active");

  DOM.rxPlantName.textContent = dx.plant_name ? `Plant: ${dx.plant_name}` : "Crop Specimen Identified";
  DOM.rxDiseaseName.textContent = dx.disease || "Pathology Assessment";

  // Severity
  const sev = (dx.severity || "moderate").toLowerCase();
  DOM.rxSeverityTag.className = `severity-tag ${sev}`;
  DOM.rxSeverityTag.textContent = sev.toUpperCase();

  // Confidence Gauge
  const confVal = Math.min(Math.max(dx.confidence || 0.85, 0), 1);
  const confPct = Math.round(confVal * 100);
  DOM.rxConfidenceText.textContent = `${confPct}%`;

  const circumference = 2 * Math.PI * 18; // r = 18
  const offset = circumference - (confVal * circumference);
  DOM.rxConfidenceProgress.style.strokeDasharray = `${circumference}`;
  DOM.rxConfidenceProgress.style.strokeDashoffset = `${offset}`;

  if (confPct >= 80) {
    DOM.rxConfidenceProgress.style.stroke = "var(--emerald-500)";
    DOM.rxConfidenceStatus.textContent = "High Confidence";
    DOM.rxConfidenceStatus.style.color = "var(--emerald-400)";
  } else if (confPct >= 60) {
    DOM.rxConfidenceProgress.style.stroke = "var(--amber-500)";
    DOM.rxConfidenceStatus.textContent = "Moderate Confidence";
    DOM.rxConfidenceStatus.style.color = "var(--amber-500)";
  } else {
    DOM.rxConfidenceProgress.style.stroke = "var(--rose-500)";
    DOM.rxConfidenceStatus.textContent = "Uncertain (Escalated)";
    DOM.rxConfidenceStatus.style.color = "var(--rose-500)";
  }

  // Cause
  DOM.rxCauseText.textContent = dx.cause || "Identified through pathognomonic visual cues and agronomic RAG retrieval.";

  // Symptoms
  renderList(DOM.rxSymptomsList, dx.symptoms, "symptom");

  // Organic Treatments
  renderList(DOM.rxOrganicList, dx.organic_treatment, "organic");

  // Chemical Treatments
  renderList(DOM.rxChemicalList, dx.chemical_treatment, "chemical");

  // Prevention
  renderList(DOM.rxPreventionList, dx.prevention, "prevention");

  // Evidence Sources
  renderList(DOM.rxEvidenceList, dx.evidence_sources, "evidence");

  // Scroll to prescription
  DOM.prescriptionPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderList(container, items, type) {
  container.innerHTML = "";
  if (!items || items.length === 0) {
    const li = document.createElement("li");
    li.className = "treatment-item";
    li.innerHTML = `<span style="color:var(--text-muted);">No specific ${type} protocols specified.</span>`;
    container.appendChild(li);
    return;
  }

  items.forEach((item, idx) => {
    const li = document.createElement("li");
    li.className = "treatment-item";
    let bulletClass = type;
    li.innerHTML = `
      <span class="treatment-bullet ${bulletClass}">${idx + 1}</span>
      <span>${escapeHtml(item)}</span>
    `;
    container.appendChild(li);
  });
}

function initTreatmentTabs() {
  const tabs = document.querySelectorAll(".treat-tab");
  const sections = document.querySelectorAll(".treatment-card-section");

  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      sections.forEach(s => s.classList.remove("active"));

      tab.classList.add("active");
      const targetId = tab.getAttribute("data-target");
      const targetSection = document.getElementById(targetId);
      if (targetSection) {
        targetSection.classList.add("active");
      }
    });
  });
}

// ==========================================================================
// Helpers & Utilities
// ==========================================================================

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function escapeForJs(str) {
  if (!str) return "";
  return String(str).replace(/'/g, "\\'").replace(/\n/g, " ");
}

function formatMarkdownish(text) {
  if (!text) return "";
  let formatted = escapeHtml(text);
  formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');
  formatted = formatted.replace(/\n\n/g, '<br><br>');
  formatted = formatted.replace(/\n/g, '<br>');
  return formatted;
}
