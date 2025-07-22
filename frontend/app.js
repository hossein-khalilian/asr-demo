
let mediaRecorder;
let audioChunks = [];
const recordBtn = document.getElementById('recordBtn');
const stopBtn = document.getElementById('stopBtn');
const transcription = document.getElementById('transcription');
const status = document.getElementById('status');
let statusInterval = null;


recordBtn.onclick = async () => {
    audioChunks = [];
    transcription.textContent = '';
    stopBtn.disabled = false;
    recordBtn.disabled = true;
    status.textContent = 'Recording...';
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        transcription.textContent = 'Error: getUserMedia not supported in this browser or context.';
        status.textContent = 'Idle';
        stopBtn.disabled = true;
        recordBtn.disabled = false;
        return;
    }
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = e => {
            if (e.data.size > 0) audioChunks.push(e.data);
        };
        mediaRecorder.onstop = async () => {
            status.textContent = 'Sending...';
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            const formData = new FormData();
            formData.append('file', audioBlob, 'recording.webm');
            try {
                const res = await fetch('http://localhost:8000/transcribe', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                transcription.textContent = data.text || 'No transcription.';
                status.textContent = 'Idle';
            } catch (err) {
                transcription.textContent = 'Error: ' + err;
                status.textContent = 'Idle';
            }
        };
        mediaRecorder.start();
    } catch (err) {
        transcription.textContent = 'Error: Could not access microphone. ' + err;
        status.textContent = 'Idle';
        stopBtn.disabled = true;
        recordBtn.disabled = false;
    }
};


stopBtn.onclick = () => {
    stopBtn.disabled = true;
    recordBtn.disabled = false;
    status.textContent = 'Processing...';
    if (mediaRecorder) {
        mediaRecorder.stop();
    } else {
        transcription.textContent = 'Error: No recording in progress.';
        status.textContent = 'Idle';
    }
};

