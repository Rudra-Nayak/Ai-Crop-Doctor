/**
 * AI Crop Doctor — Frontend Application Logic
 * Vanilla JavaScript + Bootstrap 5 + Native Web APIs
 */

(() => {
    'use strict';

    // ── Configuration ───────────────────────────────────────────────
    const API_BASE = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
        ? 'http://localhost:8000'
        : 'https://ai-crop-doctor-1u6a.onrender.com';

    // ── State ───────────────────────────────────────────────────────
    let currentCaseId = null;
    let selectedImageFile = null;
    let recordedAudioBlob = null;
    let currentLanguage = 'en';
    let mediaRecorder = null;
    let audioChunks = [];
    let recordTimerInterval = null;
    let recordStartTime = 0;
    let isProcessing = false;
    let currentAudioPlayer = null;

    // ── Translations ────────────────────────────────────────────────
    const TRANSLATIONS = {
        en: {
            welcome_title: "AI Crop Doctor",
            welcome_subtitle: "Upload a crop photo, describe the problem, or speak to the Crop Doctor.",
            cap_photo: "Crop photo analysis",
            cap_text: "Symptom description",
            cap_voice: "Voice diagnosis",
            try_asking: "Try asking:",
            input_placeholder: "Type your crop problem...",
            recording: "Recording...",
            voice_input: "Voice message",
            analyzing_1: "Analyzing your crop image...",
            analyzing_2: "Checking plant symptoms...",
            analyzing_3: "Searching agricultural knowledge base...",
            analyzing_4: "Formulating treatment recommendations...",
            ready: "Ready",
            busy: "Analyzing...",
            offline: "Offline",
            listen_audio: "Listen",
            playing_audio: "Playing...",
            stop_audio: "Stop",
            crop: "Crop",
            diagnosis: "Diagnosis",
            confidence: "Confidence",
            severity: "Severity",
            symptoms: "Symptoms",
            cause: "Cause / Pathogen",
            organic_treatment: "Organic Treatment",
            chemical_treatment: "Chemical Treatment",
            prevention: "Prevention Practices",
            evidence: "Grounded Agricultural Evidence",
            healthy_title: "Plant Appears Healthy",
            healthy_msg: "Great news! Your crop shows no clear signs of disease or severe deficiency. Continue with good agricultural practices.",
            followup_title: "More Information Needed",
            err_not_plant: "The uploaded image does not appear to be a crop or plant. Please upload a clear photo of the affected plant or leaves.",
            err_network: "Could not connect to Crop Doctor. Please check your internet connection.",
            err_server: "Crop Doctor service is temporarily unavailable. Please try again in a moment.",
            err_unsupported_img: "Please select a valid image file (JPG, PNG, WEBP, BMP under 10MB)."
        },
        hi: {
            welcome_title: "एआई फसल डॉक्टर",
            welcome_subtitle: "फसल की फोटो अपलोड करें, समस्या बताएं, या बोलकर डॉक्टर से सलाह लें।",
            cap_photo: "फोटो द्वारा रोग पहचान",
            cap_text: "लक्षणों का विवरण",
            cap_voice: "बोलकर सवाल पूछें",
            try_asking: "ये पूछकर देखें:",
            input_placeholder: "अपनी फसल की समस्या यहाँ लिखें...",
            recording: "रिकॉर्डिंग हो रही है...",
            voice_input: "आवाज़ संदेश",
            analyzing_1: "फसल की तस्वीर की जांच हो रही है...",
            analyzing_2: "पौधे के लक्षणों का मिलान किया जा रहा है...",
            analyzing_3: "कृषि ज्ञानकोष से समाधान खोजा जा रहा है...",
            analyzing_4: "उपचार और रोकथाम की सलाह तैयार हो रही है...",
            ready: "तैयार",
            busy: "जांच जारी...",
            offline: "ऑफलाइन",
            listen_audio: "सुनें",
            playing_audio: "चल रहा है...",
            stop_audio: "रोकें",
            crop: "फसल",
            diagnosis: "रोग की पहचान",
            confidence: "सटीकता",
            severity: "गंभीरता",
            symptoms: "प्रमुख लक्षण",
            cause: "रोग का कारण / रोगाणु",
            organic_treatment: "जैविक / देशी उपचार",
            chemical_treatment: "रासायनिक उपचार",
            prevention: "बचाव के उपाय",
            evidence: "प्रमाणित कृषि स्रोत",
            healthy_title: "फसल स्वस्थ दिख रही है",
            healthy_msg: "बधाई! आपकी फसल में किसी गंभीर रोग के लक्षण नहीं मिले हैं। सामान्य देखभाल जारी रखें।",
            followup_title: "अतिरिक्त जानकारी की आवश्यकता है",
            err_not_plant: "अपलोड की गई फोटो किसी फसल या पौधे की नहीं लग रही है। कृपया पत्ती या पौधे की स्पष्ट फोटो अपलोड करें।",
            err_network: "फसल डॉक्टर से संपर्क नहीं हो सका। कृपया अपना इंटरनेट कनेक्शन जांचें।",
            err_server: "सर्वर से संपर्क करने में समस्या हुई। कृपया कुछ समय बाद पुनः प्रयास करें।",
            err_unsupported_img: "कृपया सही फोटो (JPG, PNG, WEBP) अपलोड करें।"
        },
        or: {
            welcome_title: "AI ଫସଲ ଡାକ୍ତର",
            welcome_subtitle: "ଫସଲର ଫଟୋ ଅପଲୋଡ୍ କରନ୍ତୁ, ସମସ୍ୟା ଲେଖନ୍ତୁ କିମ୍ବା କହି ସମାଧାନ ପାଆନ୍ତୁ।",
            cap_photo: "ଫଟୋ ବିଶ୍ଳେଷଣ",
            cap_text: "ଲକ୍ଷଣ ବର୍ଣ୍ଣନା",
            cap_voice: "ସ୍ୱର ପରାମର୍ଶ",
            try_asking: "ଉଦାହରଣ ପ୍ରଶ୍ନ:",
            input_placeholder: "ଆପଣଙ୍କ ଫସଲ ସମସ୍ୟା ଲେଖନ୍ତୁ...",
            recording: "ରେକର୍ଡିଂ ଚାଲିଛି...",
            voice_input: "ସ୍ୱର ବାର୍ତ୍ତା",
            analyzing_1: "ଫସଲ ଫଟୋ ପରୀକ୍ଷା କରାଯାଉଛି...",
            analyzing_2: "ଲକ୍ଷଣ ଯାଞ୍ଚ ହେଉଛି...",
            analyzing_3: "କୃଷି ବିଜ୍ଞାନ ତଥ୍ୟରୁ ଉତ୍ତର ଖୋଜାଯାଉଛି...",
            analyzing_4: "ଉପଚାର ପରାମର୍ଶ ପ୍ରସ୍ତୁତ ହେଉଛି...",
            ready: "ପ୍ରସ୍ତୁତ",
            busy: "ପରୀକ୍ଷା ଚାଲିଛି...",
            offline: "ଅଫଲାଇନ୍",
            listen_audio: "ଶୁଣନ୍ତୁ",
            playing_audio: "ଚାଲିଛି...",
            stop_audio: "ବନ୍ଦ କରନ୍ତୁ",
            crop: "ଫସଲ",
            diagnosis: "ରୋଗ ନିର୍ଣ୍ଣୟ",
            confidence: "ବିଶ୍ୱସନୀୟତା",
            severity: "ତୀବ୍ରତା",
            symptoms: "ମୁଖ୍ୟ ଲକ୍ଷଣ",
            cause: "କାରଣ",
            organic_treatment: "ଜୈବିକ ଉପଚାର",
            chemical_treatment: "ରାସାୟନିକ ଉପଚାର",
            prevention: "ପ୍ରତିରୋଧ ପଦକ୍ଷେପ",
            evidence: "କୃଷି ପ୍ରମାଣ",
            healthy_title: "ଗଛ ସୁସ୍ଥ ଦେଖାଯାଉଛି",
            healthy_msg: "ଉତ୍ତମ ଖବର! ଆପଣଙ୍କ ଫସଲରେ କୌଣସି ରୋଗର ଲକ୍ଷଣ ଦେଖାଯାଉନାହିଁ।",
            followup_title: "ଅଧିକ ସୂଚନା ଆବଶ୍ୟକ",
            err_not_plant: "ଦୟାକରି ଏକ ଫସଲ କିମ୍ବା ପତ୍ରର ସ୍ପଷ୍ଟ ଫଟୋ ଅପଲୋଡ୍ କରନ୍ତୁ।",
            err_network: "ସଂଯୋଗ ବିଫଳ ହେଲା। ଇଣ୍ଟରନେଟ୍ ଯାଞ୍ଚ କରନ୍ତୁ।",
            err_server: "ଫସଲ ଡାକ୍ତର ସେବା ଅସ୍ଥାୟୀ ଭାବରେ ଅନୁପଲବ୍ଧ।",
            err_unsupported_img: "ଦୟାକରି ବୈଧ ଫଟୋ ବାଛନ୍ତୁ।"
        },
        pa: {
            welcome_title: "AI ਫ਼ਸਲ ਡਾਕਟਰ",
            welcome_subtitle: "ਫ਼ਸਲ ਦੀ ਫੋਟੋ ਅਪਲੋਡ ਕਰੋ, ਸਮੱਸਿਆ ਲਿਖੋ, ਜਾਂ ਬੋਲ ਕੇ ਸਲਾਹ ਲਵੋ।",
            cap_photo: "ਫੋਟੋ ਜਾਂਚ",
            cap_text: "ਲੱਛਣਾਂ ਦਾ ਵੇਰਵਾ",
            cap_voice: "ਬੋਲ ਕੇ ਪੁੱਛੋ",
            try_asking: "ਇਹ ਪੁੱਛ ਕੇ ਵੇਖੋ:",
            input_placeholder: "ਆਪਣੀ ਫ਼ਸਲ ਦੀ ਸਮੱਸਿਆ ਇੱਥੇ ਲਿਖੋ...",
            recording: "ਰਿਕਾਰਡਿੰਗ ਹੋ ਰਹੀ ਹੈ...",
            voice_input: "ਆਵਾਜ਼ ਸੁਨੇਹਾ",
            analyzing_1: "ਫ਼ਸਲ ਦੀ ਫੋਟੋ ਦੀ ਜਾਂਚ ਹੋ ਰਹੀ ਹੈ...",
            analyzing_2: "ਬਿਮਾਰੀ ਦੇ ਲੱਛਣ ਮਿਲਾਏ ਜਾ ਰਹੇ ਹਨ...",
            analyzing_3: "ਖੇਤੀਬਾੜੀ ਗਿਆਨਕੋਸ਼ ਵਿੱਚੋਂ ਹੱਲ ਲੱਭਿਆ ਜਾ ਰਿਹਾ ਹੈ...",
            analyzing_4: "ਇਲਾਜ ਦੀ ਸਲਾਹ ਤਿਆਰ ਹੋ ਰਹੀ ਹੈ...",
            ready: "ਤਿਆਰ",
            busy: "ਜਾਂਚ ਜਾਰੀ...",
            offline: "ਆਫਲਾਈਨ",
            listen_audio: "ਸੁਣੋ",
            playing_audio: "ਚੱਲ ਰਿਹਾ ਹੈ...",
            stop_audio: "ਰੋਕੋ",
            crop: "ਫ਼ਸਲ",
            diagnosis: "ਬਿਮਾਰੀ ਦੀ ਪਛਾਣ",
            confidence: "ਸ਼ੁੱਧਤਾ",
            severity: "ਗੰਭੀਰਤਾ",
            symptoms: "ਮੁੱਖ ਲੱਛਣ",
            cause: "ਕਾਰਨ",
            organic_treatment: "ਜੈਵਿਕ ਇਲਾਜ",
            chemical_treatment: "ਰਸਾਇਣਕ ਇਲਾਜ",
            prevention: "ਬਚਾਅ ਦੇ ਉਪਾਅ",
            evidence: "ਪ੍ਰਮਾਣਿਤ ਖੇਤੀਬਾੜੀ ਸਰੋਤ",
            healthy_title: "ਫ਼ਸਲ ਤੰਦਰੁਸਤ ਹੈ",
            healthy_msg: "ਵਧੀਆ ਗੱਲ ਹੈ! ਤੁਹਾਡੀ ਫ਼ਸਲ ਵਿੱਚ ਕੋਈ ਬਿਮਾਰੀ ਨਹੀਂ ਮਿਲੀ।",
            followup_title: "ਹੋਰ ਜਾਣਕਾਰੀ ਦੀ ਲੋੜ ਹੈ",
            err_not_plant: "ਕਿਰਪਾ ਕਰਕੇ ਕਿਸੇ ਫ਼ਸਲ ਜਾਂ ਪੱਤੇ ਦੀ ਸਾਫ਼ ਫੋਟੋ ਅਪਲੋਡ ਕਰੋ।",
            err_network: "ਕਿਰਪਾ ਕਰਕੇ ਆਪਣਾ ਇੰਟਰਨੈੱਟ ਕਨੈਕਸ਼ਨ ਚੈੱਕ ਕਰੋ।",
            err_server: "ਸੇਵਾ ਅਸਥਾਈ ਤੌਰ 'ਤੇ ਉਪਲਬਧ ਨਹੀਂ ਹੈ।",
            err_unsupported_img: "ਕਿਰਪਾ ਕਰਕੇ ਸਹੀ ਫੋਟੋ ਅਪਲੋਡ ਕਰੋ।"
        }
    };

    function t(key) {
        return (TRANSLATIONS[currentLanguage] && TRANSLATIONS[currentLanguage][key]) ||
               (TRANSLATIONS.en && TRANSLATIONS.en[key]) || key;
    }

    // ── DOM Elements ────────────────────────────────────────────────
    const topNav = document.getElementById('topNav');
    const statusText = document.getElementById('statusText');
    const langBtn = document.getElementById('langBtn');
    const langLabel = document.getElementById('langLabel');
    const langSelector = document.getElementById('langSelector');
    const langDropdown = document.getElementById('langDropdown');
    const langOptions = document.querySelectorAll('.lang-option');

    const chatContainer = document.getElementById('chatContainer');
    const welcomeScreen = document.getElementById('welcomeScreen');
    const welcomeSuggestions = document.getElementById('welcomeSuggestions');
    const messagesList = document.getElementById('messagesList');

    const imageInput = document.getElementById('imageInput');
    const attachBtn = document.getElementById('attachBtn');
    const messageInput = document.getElementById('messageInput');
    const micBtn = document.getElementById('micBtn');
    const sendBtn = document.getElementById('sendBtn');

    const imagePreviewBar = document.getElementById('imagePreviewBar');
    const imagePreviewImg = document.getElementById('imagePreviewImg');
    const imageRemoveBtn = document.getElementById('imageRemoveBtn');

    const recordingBar = document.getElementById('recordingBar');
    const recTimer = document.getElementById('recTimer');
    const recCancelBtn = document.getElementById('recCancelBtn');
    const recStopBtn = document.getElementById('recStopBtn');

    // ── Initialization ──────────────────────────────────────────────
    document.addEventListener('DOMContentLoaded', () => {
        setupEventListeners();
        checkBackendHealth();
        applyLanguage(currentLanguage);
        autoResizeTextarea();
    });

    // ── Health Check ────────────────────────────────────────────────
    async function checkBackendHealth() {
        try {
            const res = await fetch(`${API_BASE}/api/health`, { method: 'GET' });
            if (res.ok) {
                const data = await res.json();
                if (data.status === 'ok') {
                    setStatus('ready');
                } else {
                    setStatus('ready');
                }
            } else {
                setStatus('offline');
            }
        } catch {
            setStatus('offline');
        }
    }

    function setStatus(state) {
        const dot = document.querySelector('.status-dot');
        if (state === 'ready') {
            statusText.textContent = t('ready');
            if (dot) dot.style.background = 'var(--brand-500)';
        } else if (state === 'busy') {
            statusText.textContent = t('busy');
            if (dot) dot.style.background = 'var(--warning)';
        } else {
            statusText.textContent = t('offline');
            if (dot) dot.style.background = 'var(--danger)';
        }
    }

    // ── Event Listeners ─────────────────────────────────────────────
    function setupEventListeners() {
        // Language selector
        langBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            langSelector.classList.toggle('open');
        });

        document.addEventListener('click', () => {
            langSelector.classList.remove('open');
        });

        langOptions.forEach(opt => {
            opt.addEventListener('click', (e) => {
                e.stopPropagation();
                const lang = opt.getAttribute('data-lang');
                const label = opt.getAttribute('data-label');
                selectLanguage(lang, label);
                langSelector.classList.remove('open');
            });
        });

        // Suggestion chips
        welcomeSuggestions.addEventListener('click', (e) => {
            const chip = e.target.closest('.suggestion-chip');
            if (!chip) return;
            const text = chip.getAttribute('data-text');
            if (text) {
                messageInput.value = text;
                updateSendState();
                handleSend();
            }
        });

        // Image attachment
        attachBtn.addEventListener('click', () => {
            imageInput.click();
        });

        imageInput.addEventListener('change', handleImageSelect);
        imageRemoveBtn.addEventListener('click', removeSelectedImage);

        // Text input handling
        messageInput.addEventListener('input', () => {
            autoResizeTextarea();
            updateSendState();
        });

        messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (!sendBtn.disabled && !isProcessing) {
                    handleSend();
                }
            }
        });

        // Voice recording
        micBtn.addEventListener('click', toggleRecording);
        recCancelBtn.addEventListener('click', cancelRecording);
        recStopBtn.addEventListener('click', stopRecording);

        // Send submission
        sendBtn.addEventListener('click', () => {
            if (!sendBtn.disabled && !isProcessing) {
                handleSend();
            }
        });
    }

    // ── Language Handling ───────────────────────────────────────────
    function selectLanguage(lang, label) {
        currentLanguage = lang;
        langLabel.textContent = label;

        langOptions.forEach(opt => {
            if (opt.getAttribute('data-lang') === lang) {
                opt.classList.add('active');
            } else {
                opt.classList.remove('active');
            }
        });

        applyLanguage(lang);
    }

    function applyLanguage(lang) {
        // Update all data-i18n elements
        document.querySelectorAll('[data-i18n]').forEach(el => {
            const key = el.getAttribute('data-i18n');
            el.textContent = t(key);
        });

        // Update placeholders
        document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
            const key = el.getAttribute('data-i18n-placeholder');
            el.setAttribute('placeholder', t(key));
        });

        // Update status text if not busy
        if (!isProcessing) {
            statusText.textContent = t('ready');
        }
    }

    // ── Textarea Auto-resize ────────────────────────────────────────
    function autoResizeTextarea() {
        messageInput.style.height = 'auto';
        const newHeight = Math.min(messageInput.scrollHeight, 120);
        messageInput.style.height = `${newHeight}px`;
    }

    // ── Input State & Validation ────────────────────────────────────
    function updateSendState() {
        const hasText = messageInput.value.trim().length > 0;
        const hasImage = !!selectedImageFile;
        const hasAudio = !!recordedAudioBlob;

        if ((hasText || hasImage || hasAudio) && !isProcessing) {
            sendBtn.removeAttribute('disabled');
            sendBtn.classList.add('active');
        } else {
            sendBtn.setAttribute('disabled', 'true');
            sendBtn.classList.remove('active');
        }
    }

    // ── Image Handling ──────────────────────────────────────────────
    function handleImageSelect(e) {
        const file = e.target.files && e.target.files[0];
        if (!file) return;

        // Size check (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            alert(t('err_unsupported_img'));
            imageInput.value = '';
            return;
        }

        selectedImageFile = file;

        // Display preview
        const reader = new FileReader();
        reader.onload = (loadEvt) => {
            imagePreviewImg.src = loadEvt.target.result;
            imagePreviewBar.classList.add('visible');
            attachBtn.classList.add('has-image');
            updateSendState();
            messageInput.focus();
        };
        reader.readAsDataURL(file);
    }

    function removeSelectedImage() {
        selectedImageFile = null;
        imageInput.value = '';
        imagePreviewImg.src = '';
        imagePreviewBar.classList.remove('visible');
        attachBtn.classList.remove('has-image');
        updateSendState();
    }

    // ── Real Voice Recording via MediaRecorder ──────────────────────
    async function toggleRecording() {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            stopRecording();
        } else {
            startRecording();
        }
    }

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioChunks = [];

            // Select supported mimeType
            let mimeType = 'audio/webm';
            if (!MediaRecorder.isTypeSupported('audio/webm')) {
                if (MediaRecorder.isTypeSupported('audio/mp4')) {
                    mimeType = 'audio/mp4';
                } else if (MediaRecorder.isTypeSupported('audio/ogg')) {
                    mimeType = 'audio/ogg';
                } else {
                    mimeType = '';
                }
            }

            mediaRecorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);

            mediaRecorder.ondataavailable = (e) => {
                if (e.data && e.data.size > 0) {
                    audioChunks.push(e.data);
                }
            };

            mediaRecorder.onstop = () => {
                const finalType = mediaRecorder.mimeType || 'audio/webm';
                recordedAudioBlob = new Blob(audioChunks, { type: finalType });
                stream.getTracks().forEach(track => track.stop());
                clearInterval(recordTimerInterval);
                recordingBar.classList.remove('visible');
                micBtn.classList.remove('recording');

                // Transcribe audio using backend STT to show immediate preview text if textarea empty
                transcribeAndSetText(recordedAudioBlob);
                updateSendState();
            };

            mediaRecorder.start(250);
            recordStartTime = Date.now();
            updateTimerDisplay();
            recordTimerInterval = setInterval(updateTimerDisplay, 500);

            recordingBar.classList.add('visible');
            micBtn.classList.add('recording');

        } catch (err) {
            console.error('Microphone access denied or error:', err);
            alert("Microphone permission is required to record voice symptoms.");
        }
    }

    function updateTimerDisplay() {
        const elapsedSec = Math.floor((Date.now() - recordStartTime) / 1000);
        const mins = Math.floor(elapsedSec / 60);
        const secs = elapsedSec % 60;
        recTimer.textContent = `${mins}:${secs < 10 ? '0' : ''}${secs}`;
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
        }
    }

    function cancelRecording() {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.onstop = () => {
                const stream = mediaRecorder.stream;
                if (stream) stream.getTracks().forEach(track => track.stop());
                clearInterval(recordTimerInterval);
                recordingBar.classList.remove('visible');
                micBtn.classList.remove('recording');
                recordedAudioBlob = null;
                audioChunks = [];
                updateSendState();
            };
            mediaRecorder.stop();
        } else {
            recordingBar.classList.remove('visible');
            micBtn.classList.remove('recording');
            recordedAudioBlob = null;
            audioChunks = [];
            updateSendState();
        }
    }

    async function transcribeAndSetText(blob) {
        try {
            const formData = new FormData();
            const ext = blob.type.includes('mp4') ? 'mp4' : (blob.type.includes('ogg') ? 'ogg' : 'webm');
            formData.append('audio', blob, `voice_input.${ext}`);

            const res = await fetch(`${API_BASE}/api/voice/transcribe`, {
                method: 'POST',
                body: formData
            });

            if (res.ok) {
                const data = await res.json();
                if (data && data.text && !messageInput.value.trim()) {
                    messageInput.value = data.text;
                    autoResizeTextarea();
                    updateSendState();
                }
            }
        } catch (e) {
            console.warn("Background transcription error:", e);
        }
    }

    // ── Send & Submission Flow ──────────────────────────────────────
    async function handleSend() {
        const userText = messageInput.value.trim();
        const imageFile = selectedImageFile;
        const audioBlob = recordedAudioBlob;

        if (!userText && !imageFile && !audioBlob) return;

        // Hide welcome screen, show messages list
        welcomeScreen.classList.add('hidden');
        messagesList.classList.add('active');

        // Render user message card
        renderUserMessage({
            text: userText,
            imageFile: imageFile,
            hasVoice: !!audioBlob
        });

        // Clear composer state immediately
        messageInput.value = '';
        autoResizeTextarea();
        removeSelectedImage();
        recordedAudioBlob = null;
        updateSendState();

        // Lock interface & show animated loading step
        isProcessing = true;
        setStatus('busy');
        const loadingCard = showLoadingIndicator();
        scrollToBottom();

        try {
            // Construct multipart/form-data for /api/diagnosis
            const formData = new FormData();

            if (imageFile) {
                formData.append('image', imageFile, imageFile.name);
            }
            if (userText) {
                formData.append('text', userText);
            }
            if (audioBlob) {
                const ext = audioBlob.type.includes('mp4') ? 'mp4' : (audioBlob.type.includes('ogg') ? 'ogg' : 'webm');
                formData.append('audio', audioBlob, `speech.${ext}`);
            }
            if (currentCaseId) {
                formData.append('case_id', currentCaseId);
            }

            // Notice: We NEVER set Content-Type header manually; fetch sets multipart boundary automatically!
            const res = await fetch(`${API_BASE}/api/diagnosis`, {
                method: 'POST',
                body: formData
            });

            removeLoadingIndicator(loadingCard);

            if (!res.ok) {
                renderErrorCard(t('err_server'));
                return;
            }

            const data = await res.json();

            // Track conversation case_id
            if (data.case_id) {
                currentCaseId = data.case_id;
            }

            // Render AI diagnosis response
            renderAiMessage(data);

        } catch (err) {
            console.error('Diagnosis request error:', err);
            removeLoadingIndicator(loadingCard);
            renderErrorCard(t('err_network'));
        } finally {
            isProcessing = false;
            setStatus('ready');
            updateSendState();
            scrollToBottom();
        }
    }

    // ── Message Renderers ───────────────────────────────────────────
    function renderUserMessage({ text, imageFile, hasVoice }) {
        const tmpl = document.getElementById('tmplUserMsg');
        const clone = tmpl.content.cloneNode(true);

        const imgContainer = clone.querySelector('[data-slot="userImage"]');
        const textContainer = clone.querySelector('[data-slot="userText"]');
        const voiceBadge = clone.querySelector('[data-slot="voiceBadge"]');

        if (imageFile) {
            const img = document.createElement('img');
            img.src = URL.createObjectURL(imageFile);
            img.alt = "Uploaded crop";
            imgContainer.appendChild(img);
            imgContainer.classList.add('has-image');
        }

        if (text) {
            textContainer.textContent = text;
        } else if (!imageFile && hasVoice) {
            textContainer.textContent = t('voice_input');
        }

        if (hasVoice) {
            voiceBadge.classList.add('visible');
        }

        messagesList.appendChild(clone);
    }

    function showLoadingIndicator() {
        const tmpl = document.getElementById('tmplLoading');
        const clone = tmpl.content.cloneNode(true);
        const node = clone.querySelector('.msg-loading');
        messagesList.appendChild(clone);

        // Cycle through friendly status messages
        const textEl = node.querySelector('#loadingText');
        const stages = [t('analyzing_1'), t('analyzing_2'), t('analyzing_3'), t('analyzing_4')];
        let idx = 0;
        const interval = setInterval(() => {
            idx = (idx + 1) % stages.length;
            if (textEl) textEl.textContent = stages[idx];
        }, 1500);

        node._interval = interval;
        return node;
    }

    function removeLoadingIndicator(node) {
        if (node) {
            if (node._interval) clearInterval(node._interval);
            node.remove();
        }
    }

    function renderAiMessage(data) {
        const tmpl = document.getElementById('tmplAiMsg');
        const clone = tmpl.content.cloneNode(true);
        const contentBox = clone.querySelector('[data-slot="aiContent"]');

        const { response_text, diagnosis, needs_followup, followup_question, confidence } = data;

        // 1. Conversational Response Text
        if (response_text) {
            const textDiv = document.createElement('div');
            textDiv.className = 'ai-response-text';
            textDiv.textContent = response_text;
            contentBox.appendChild(textDiv);
        }

        // 2. TTS Voice Playback Button
        if (response_text && response_text.length > 5) {
            const ttsBtn = document.createElement('button');
            ttsBtn.className = 'tts-play-btn';
            ttsBtn.innerHTML = `<i class="bi bi-volume-up"></i> <span>${t('listen_audio')}</span>`;
            ttsBtn.addEventListener('click', () => handleTtsPlayback(response_text, ttsBtn));
            contentBox.appendChild(ttsBtn);
        }

        // 3. Check for specific conditions
        if (needs_followup && followup_question) {
            // Follow-up required state
            const fCard = document.createElement('div');
            fCard.className = 'followup-card';
            fCard.innerHTML = `
                <div class="followup-title"><i class="bi bi-question-circle-fill"></i> ${t('followup_title')}</div>
                <div class="followup-question">${escapeHtml(followup_question)}</div>
            `;
            contentBox.appendChild(fCard);

        } else if (diagnosis) {
            const isHealthy = (diagnosis.disease && diagnosis.disease.toLowerCase().includes('healthy'));

            if (isHealthy) {
                // Healthy Plant State
                const hCard = document.createElement('div');
                hCard.className = 'healthy-card';
                hCard.innerHTML = `
                    <div class="healthy-icon"><i class="bi bi-check-circle-fill"></i></div>
                    <div class="healthy-title">${t('healthy_title')}</div>
                    <div class="healthy-text">${t('healthy_msg')}</div>
                `;
                contentBox.appendChild(hCard);

            } else {
                // Full Structured Diagnosis Card
                const dCard = buildDiagnosisCard(diagnosis, confidence);
                contentBox.appendChild(dCard);
            }
        }

        messagesList.appendChild(clone);
    }

    function buildDiagnosisCard(diag, confidenceScore) {
        const card = document.createElement('div');
        card.className = 'diagnosis-card';

        const confVal = Math.round(((diag.confidence || confidenceScore || 0)) * 100);
        let confClass = 'diag-badge-confidence';
        let fillClass = 'confidence-fill';
        if (confVal < 60) {
            confClass += ' low';
            fillClass += ' low';
        } else if (confVal < 80) {
            confClass += ' medium';
            fillClass += ' medium';
        }

        const severity = (diag.severity || 'Moderate').toLowerCase();
        let sevClass = `diag-badge-severity ${severity}`;

        // Header
        let html = `
            <div class="diag-header">
                <div class="diag-header-left">
                    <span class="diag-plant"><i class="bi bi-flower2"></i> ${escapeHtml(diag.plant_name || 'Crop')}</span>
                    <span class="diag-disease">${escapeHtml(diag.disease || 'Plant Condition')}</span>
                </div>
                <div class="diag-badges">
                    <span class="diag-badge ${confClass}"><i class="bi bi-shield-check"></i> ${confVal}% ${t('confidence')}</span>
                    ${diag.severity ? `<span class="diag-badge ${sevClass}"><i class="bi bi-exclamation-triangle"></i> ${escapeHtml(diag.severity)}</span>` : ''}
                </div>
            </div>
            <div class="confidence-bar-wrap">
                <div class="confidence-bar-label">
                    <span>${t('confidence')}</span>
                    <span>${confVal}%</span>
                </div>
                <div class="confidence-bar">
                    <div class="${fillClass}" style="width: ${confVal}%"></div>
                </div>
            </div>
            <div class="diag-sections">
        `;

        // Cause / Pathogen
        if (diag.cause) {
            html += `
                <div class="diag-section">
                    <div class="diag-section-title"><i class="bi bi-bug"></i> ${t('cause')}</div>
                    <div class="diag-cause">${escapeHtml(diag.cause)}</div>
                </div>
            `;
        }

        // Symptoms
        if (diag.symptoms && diag.symptoms.length > 0) {
            html += `
                <div class="diag-section">
                    <div class="diag-section-title"><i class="bi bi-card-checklist"></i> ${t('symptoms')}</div>
                    <ul class="diag-list">
                        ${diag.symptoms.map(s => `<li>${escapeHtml(s)}</li>`).join('')}
                    </ul>
                </div>
            `;
        }

        // Organic Treatment
        if (diag.organic_treatment && diag.organic_treatment.length > 0) {
            html += `
                <div class="diag-section">
                    <div class="diag-section-title"><i class="bi bi-tree"></i> ${t('organic_treatment')}</div>
                    <ul class="diag-list">
                        ${diag.organic_treatment.map(o => `<li>${escapeHtml(o)}</li>`).join('')}
                    </ul>
                </div>
            `;
        }

        // Chemical Treatment
        if (diag.chemical_treatment && diag.chemical_treatment.length > 0) {
            html += `
                <div class="diag-section">
                    <div class="diag-section-title"><i class="bi bi-capsule"></i> ${t('chemical_treatment')}</div>
                    <ul class="diag-list">
                        ${diag.chemical_treatment.map(c => `<li>${escapeHtml(c)}</li>`).join('')}
                    </ul>
                </div>
            `;
        }

        // Prevention
        if (diag.prevention && diag.prevention.length > 0) {
            html += `
                <div class="diag-section">
                    <div class="diag-section-title"><i class="bi bi-shield-plus"></i> ${t('prevention')}</div>
                    <ul class="diag-list">
                        ${diag.prevention.map(p => `<li>${escapeHtml(p)}</li>`).join('')}
                    </ul>
                </div>
            `;
        }

        // Evidence Sources from Supabase pgvector RAG
        if (diag.evidence_sources && diag.evidence_sources.length > 0) {
            html += `
                <div class="diag-section">
                    <div class="diag-section-title"><i class="bi bi-journal-bookmark"></i> ${t('evidence')}</div>
                    <ul class="diag-list evidence-list">
                        ${diag.evidence_sources.map(e => `<li><i class="bi bi-file-earmark-text"></i> ${escapeHtml(e)}</li>`).join('')}
                    </ul>
                </div>
            `;
        }

        html += `</div>`;
        card.innerHTML = html;
        return card;
    }

    function renderErrorCard(message) {
        const tmpl = document.getElementById('tmplAiMsg');
        const clone = tmpl.content.cloneNode(true);
        const contentBox = clone.querySelector('[data-slot="aiContent"]');

        const errDiv = document.createElement('div');
        errDiv.className = 'error-card';
        errDiv.innerHTML = `
            <div class="error-title"><i class="bi bi-exclamation-octagon-fill"></i> ${t('welcome_title')}</div>
            <div class="error-text">${escapeHtml(message)}</div>
        `;

        contentBox.appendChild(errDiv);
        messagesList.appendChild(clone);
    }

    // ── TTS Synthesis & Audio Playback ──────────────────────────────
    async function handleTtsPlayback(text, btn) {
        if (currentAudioPlayer) {
            currentAudioPlayer.pause();
            currentAudioPlayer = null;
            document.querySelectorAll('.tts-play-btn').forEach(b => {
                b.classList.remove('playing');
                b.innerHTML = `<i class="bi bi-volume-up"></i> <span>${t('listen_audio')}</span>`;
            });
            return;
        }

        btn.classList.add('playing');
        btn.innerHTML = `<i class="bi bi-stop-circle"></i> <span>${t('playing_audio')}</span>`;

        try {
            const formData = new FormData();
            formData.append('text', text);

            const res = await fetch(`${API_BASE}/api/voice/synthesize`, {
                method: 'POST',
                body: formData
            });

            if (!res.ok || res.headers.get('X-TTS-Failed') === 'true') {
                // Fallback to browser SpeechSynthesis API if backend TTS unavailable
                speakWithBrowser(text, btn);
                return;
            }

            const audioBlob = await res.blob();
            if (audioBlob.size < 100) {
                speakWithBrowser(text, btn);
                return;
            }

            const audioUrl = URL.createObjectURL(audioBlob);
            const audio = new Audio(audioUrl);
            currentAudioPlayer = audio;

            audio.onended = () => {
                btn.classList.remove('playing');
                btn.innerHTML = `<i class="bi bi-volume-up"></i> <span>${t('listen_audio')}</span>`;
                currentAudioPlayer = null;
            };

            audio.onerror = () => {
                speakWithBrowser(text, btn);
            };

            await audio.play();

        } catch (e) {
            console.warn("Backend TTS failed, using browser synthesis fallback:", e);
            speakWithBrowser(text, btn);
        }
    }

    function speakWithBrowser(text, btn) {
        if (!('speechSynthesis' in window)) {
            btn.classList.remove('playing');
            btn.innerHTML = `<i class="bi bi-volume-up"></i> <span>${t('listen_audio')}</span>`;
            return;
        }

        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        if (currentLanguage === 'hi') utterance.lang = 'hi-IN';
        else if (currentLanguage === 'pa') utterance.lang = 'pa-IN';
        else utterance.lang = 'en-US';

        utterance.onend = () => {
            btn.classList.remove('playing');
            btn.innerHTML = `<i class="bi bi-volume-up"></i> <span>${t('listen_audio')}</span>`;
            currentAudioPlayer = null;
        };

        utterance.onerror = () => {
            btn.classList.remove('playing');
            btn.innerHTML = `<i class="bi bi-volume-up"></i> <span>${t('listen_audio')}</span>`;
            currentAudioPlayer = null;
        };

        window.speechSynthesis.speak(utterance);
    }

    // ── Utilities ───────────────────────────────────────────────────
    function scrollToBottom() {
        requestAnimationFrame(() => {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        });
    }

    function escapeHtml(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

})();
