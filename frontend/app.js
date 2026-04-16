document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const audioPreview = document.getElementById('audio-preview');
    const filenameDisplay = document.getElementById('filename-display');
    const audioPlayer = document.getElementById('audio-player');
    const resetBtn = document.getElementById('reset-btn');
    
    const loadingState = document.getElementById('loading-state');
    const resultsDashboard = document.getElementById('results-dashboard');
    
    // Results Elements
    const spectrogramGraph = document.getElementById('spectrogram-graph');
    const normalPct = document.getElementById('normal-pct');
    const anomalyPct = document.getElementById('anomaly-pct');
    const normalBar = document.getElementById('normal-bar');
    const anomalyBar = document.getElementById('anomaly-bar');
    const alertBox = document.getElementById('alert-box');

    // Drag and drop setup
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    ['dragleave', 'dragend'].forEach(type => {
        dropZone.addEventListener(type, () => {
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    resetBtn.addEventListener('click', () => {
        audioPlayer.pause();
        dropZone.classList.remove('hidden');
        audioPreview.classList.add('hidden');
        resultsDashboard.classList.add('hidden');
        fileInput.value = '';
    });

    function handleFile(file) {
        if (!file.name.endsWith('.wav')) {
            alert('Please upload a .wav file.');
            return;
        }

        // Setup File preview
        filenameDisplay.textContent = file.name;
        audioPlayer.src = URL.createObjectURL(file);
        
        dropZone.classList.add('hidden');
        audioPreview.classList.remove('hidden');
        resultsDashboard.classList.add('hidden');
        loadingState.classList.remove('hidden');

        // Send to FastAPI Backend
        uploadAndAnalyze(file);
    }

    async function uploadAndAnalyze(file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            // Note: Update URL if hosted externally. Assuming localhost for local testing.
            const response = await fetch('http://localhost:8000/api/analyze', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Server error occurred');
            }

            const data = await response.json();
            renderResults(data);

        } catch (error) {
            console.error('Error during analysis:', error);
            alert(`Analysis failed: ${error.message}`);
        } finally {
            loadingState.classList.add('hidden');
        }
    }

    function renderResults(data) {
        // Render Spectrogram
        spectrogramGraph.src = data.spectrogram_b64;

        // Render Bars & Pct
        setTimeout(() => {
            normalBar.style.width = `${data.normal_probability}%`;
            anomalyBar.style.width = `${data.anomaly_probability}%`;
        }, 100); // Slight delay for CSS animation to trigger

        normalPct.textContent = `${data.normal_probability.toFixed(2)}%`;
        anomalyPct.textContent = `${data.anomaly_probability.toFixed(2)}%`;

        // Configure Alert Box
        if (data.is_anomaly) {
            alertBox.className = 'alert-box danger';
            alertBox.innerHTML = `<i>!</i><span>🚨 CRITICAL ANOMALY DETECTED</span>`;
        } else {
            alertBox.className = 'alert-box success';
            alertBox.innerHTML = `<i>✓</i><span>Audio is completely normal.</span>`;
        }

        resultsDashboard.classList.remove('hidden');
    }
});
