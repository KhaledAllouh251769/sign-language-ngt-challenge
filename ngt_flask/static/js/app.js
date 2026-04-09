'use strict';

// ── State ─────────────────────────────────────────────────────────────────────
const state = {
  sentence:      [],
  targetLetter:  null,
  mirror:        false,
  streak:        0,
  STREAK_NEEDED: 20,
  practiceActive: false,
};

const DYNAMIC_LETTERS = new Set(['H','J','U','X','Z']);
const ALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

// ── MediaPipe Hands ───────────────────────────────────────────────────────────
let handsInstance = null;
let currentLandmarks = null;

function initHands(videoEl, onResults) {
  if (handsInstance) { handsInstance.close(); handsInstance = null; }

  handsInstance = new Hands({
    locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`
  });
  handsInstance.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence:  0.7,
  });
  handsInstance.onResults(onResults);

  const cam = new Camera(videoEl, {
    onFrame: async () => { await handsInstance.send({ image: videoEl }); },
    width: 640, height: 480,
  });
  cam.start();
  return cam;
}

// ── Extract landmarks from MediaPipe result ───────────────────────────────────
function extractLandmarks(results, mirror) {
  if (!results.multiHandLandmarks || !results.multiHandLandmarks.length) return null;
  return results.multiHandLandmarks[0].map(lm => {
    const x = mirror ? 1.0 - lm.x : lm.x;
    return [x, lm.y, lm.z];
  });
}

// ── Classify via Flask ────────────────────────────────────────────────────────
async function classify(landmarks) {
  try {
    const res = await fetch('/classify', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ landmarks }),
    });
    return await res.json();   // { letter, confidence }
  } catch {
    return { letter: null, confidence: 0 };
  }
}

// ── Draw hand skeleton on canvas ──────────────────────────────────────────────
const CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
];

function drawHand(canvas, landmarks, color = '#7c6fff') {
  const ctx   = canvas.getContext('2d');
  const w     = canvas.width;
  const h     = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!landmarks) return;

  ctx.strokeStyle = color;
  ctx.lineWidth   = 2;
  for (const [a, b] of CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(landmarks[a][0] * w, landmarks[a][1] * h);
    ctx.lineTo(landmarks[b][0] * w, landmarks[b][1] * h);
    ctx.stroke();
  }
  ctx.fillStyle = '#fff';
  for (const [x, y] of landmarks) {
    ctx.beginPath();
    ctx.arc(x * w, y * h, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

// ── Sentence helpers ──────────────────────────────────────────────────────────
function renderSentence() {
  const box = document.getElementById('sentenceBox');
  box.innerHTML = '';
  if (!state.sentence.length) {
    box.innerHTML = '<span class="sentence-placeholder">Start signing...</span>';
    return;
  }
  for (const ch of state.sentence) {
    if (ch === ' ') {
      const sp = document.createElement('span');
      sp.style.display = 'inline-block';
      sp.style.width   = '20px';
      box.appendChild(sp);
    } else {
      const chip = document.createElement('span');
      chip.className   = 'letter-chip';
      chip.textContent = ch;
      box.appendChild(chip);
    }
  }
}

function addLetter(letter) {
  state.sentence.push(letter);
  renderSentence();
  document.getElementById('feedbackAdded').textContent = letter;
}

// ── Keyboard builder ──────────────────────────────────────────────────────────
function buildKeyboard() {
  const rows = ['QWERTYUIOP', 'ASDFGHJKL', 'ZXCVBNM'];
  rows.forEach((row, i) => {
    const rowEl = document.querySelector(`.kb-row[data-row="${row}"]`);
    for (const letter of row) {
      const btn = document.createElement('button');
      btn.className       = 'kb-key';
      btn.textContent     = letter;
      btn.dataset.letter  = letter;
      btn.addEventListener('click', () => selectLetter(letter));
      rowEl.appendChild(btn);
    }
  });
}

// ── Select target letter → open camera automatically ─────────────────────────
let sentenceCam = null;

function selectLetter(letter) {
  state.targetLetter = letter;
  state.streak       = 0;

  // Highlight key
  document.querySelectorAll('.kb-key').forEach(k => k.classList.remove('active'));
  document.querySelector(`.kb-key[data-letter="${letter}"]`).classList.add('active');

  // Show camera panel + target badge
  document.getElementById('cameraPanel').style.display = 'block';
  const badge = document.getElementById('targetBadge');
  badge.textContent = letter;
  badge.classList.remove('hidden');

  // Reset progress & feedback
  document.getElementById('progressBar').style.width = '0%';
  document.getElementById('feedbackLetter').textContent = '—';
  document.getElementById('feedbackLetter').className   = 'feedback-letter';
  document.getElementById('feedbackConf').textContent   = '—';

  // Start camera if not running
  if (!sentenceCam) {
    const video   = document.getElementById('video');
    const overlay = document.getElementById('overlay');

    sentenceCam = initHands(video, async (results) => {
      const lms = extractLandmarks(results, state.mirror);
      currentLandmarks = lms;

      // Sync canvas size
      overlay.width  = video.videoWidth  || 640;
      overlay.height = video.videoHeight || 480;
      drawHand(overlay, lms);

      if (!state.targetLetter) return;

      if (lms) {
        const { letter, confidence } = await classify(lms);
        updateFeedback(letter, confidence);

        if (letter === state.targetLetter && confidence > 0.5) {
          state.streak++;
        } else {
          state.streak = Math.max(0, state.streak - 2);
        }

        const frac = Math.min(state.streak / state.STREAK_NEEDED, 1.0);
        document.getElementById('progressBar').style.width = (frac * 100) + '%';

        if (state.streak >= state.STREAK_NEEDED) {
          // ✅ Add letter!
          addLetter(state.targetLetter);
          state.targetLetter = null;
          state.streak       = 0;
          document.getElementById('progressBar').style.width = '0%';
          document.querySelectorAll('.kb-key').forEach(k => k.classList.remove('active'));
          badge.classList.add('hidden');
        }
      } else {
        state.streak = 0;
        document.getElementById('progressBar').style.width = '0%';
        document.getElementById('feedbackLetter').textContent = '—';
        document.getElementById('feedbackLetter').className   = 'feedback-letter';
        document.getElementById('feedbackConf').textContent   = '—';
      }
    });
  }
}

function updateFeedback(letter, confidence) {
  const letterEl = document.getElementById('feedbackLetter');
  const confEl   = document.getElementById('feedbackConf');

  letterEl.textContent = letter || '—';
  confEl.textContent   = letter ? Math.round(confidence * 100) + '%' : '—';

  letterEl.className = 'feedback-letter';
  if (letter && state.targetLetter) {
    if (letter === state.targetLetter && confidence > 0.7) letterEl.classList.add('correct');
    else if (confidence > 0.5)                              letterEl.classList.add('warn');
    else                                                    letterEl.classList.add('danger');
  }
}

// ── Control buttons ───────────────────────────────────────────────────────────
document.getElementById('btnSpace').addEventListener('click', () => {
  state.sentence.push(' ');
  renderSentence();
});
document.getElementById('btnDelete').addEventListener('click', () => {
  state.sentence.pop();
  renderSentence();
});
document.getElementById('btnClear').addEventListener('click', () => {
  state.sentence = [];
  renderSentence();
  document.getElementById('feedbackAdded').textContent = '—';
});
document.getElementById('btnCopy').addEventListener('click', () => {
  const text = state.sentence.join('');
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById('btnCopy');
    btn.textContent = '✅ Copied!';
    setTimeout(() => { btn.textContent = '📋 Copy'; }, 1500);
  });
});

// ── Practice mode ─────────────────────────────────────────────────────────────
let practiceCam = null;

document.getElementById('practiceStartBtn').addEventListener('click', () => {
  document.getElementById('practiceStartBtn').style.display = 'none';
  document.getElementById('practiceStopBtn').style.display  = 'inline-flex';

  const video   = document.getElementById('practiceVideo');
  const overlay = document.getElementById('practiceOverlay');

  practiceCam = initHands(video, async (results) => {
    const lms = extractLandmarks(results, state.mirror);
    overlay.width  = video.videoWidth  || 640;
    overlay.height = video.videoHeight || 480;
    drawHand(overlay, lms);

    const letterEl  = document.getElementById('practiceLetter');
    const confBar   = document.getElementById('practiceConfBar');
    const confLabel = document.getElementById('practiceConfLabel');

    if (lms) {
      const { letter, confidence } = await classify(lms);
      letterEl.textContent          = letter || '—';
      confBar.style.width           = Math.round((confidence || 0) * 100) + '%';
      confLabel.textContent         = letter
        ? `${Math.round(confidence * 100)}% confidence`
        : 'Show a sign';
    } else {
      letterEl.textContent  = '—';
      confBar.style.width   = '0%';
      confLabel.textContent = 'No hand detected';
    }
  });
});

document.getElementById('practiceStopBtn').addEventListener('click', () => {
  if (practiceCam) { practiceCam.stop(); practiceCam = null; }
  document.getElementById('practiceStartBtn').style.display = 'inline-flex';
  document.getElementById('practiceStopBtn').style.display  = 'none';
  document.getElementById('practiceLetter').textContent     = '—';
  document.getElementById('practiceConfBar').style.width    = '0%';
  document.getElementById('practiceConfLabel').textContent  = 'Show a sign';
});

// ── Learn mode ────────────────────────────────────────────────────────────────
const LEARN_DESCRIPTIONS = {
  A:'Make a fist with thumb on side.',  B:'Hold fingers straight up, thumb folded.',
  C:'Curve hand into a C shape.',       D:'Index points up, other fingers and thumb form a circle.',
  E:'Curl all fingers down.',           F:'Touch index to thumb, other fingers up.',
  G:'Point index finger sideways.',     H:'Two fingers pointing sideways.',
  I:'Pinky finger up.',                 J:'Pinky up, draw a J in the air.',
  K:'Index and middle up, thumb between them.', L:'L-shape with index and thumb.',
  M:'Three fingers over thumb.',        N:'Two fingers over thumb.',
  O:'All fingers and thumb form an O.', P:'Like K but pointing down.',
  Q:'Like G but pointing down.',        R:'Cross index and middle finger.',
  S:'Fist with thumb over fingers.',    T:'Thumb between index and middle.',
  U:'Index and middle up together.',    V:'Index and middle in a V.',
  W:'Index, middle, ring fingers up.',  X:'Hook index finger.',
  Y:'Thumb and pinky out.',             Z:'Draw a Z with index finger.',
};

function buildLearn() {
  const container = document.getElementById('learnAlphabet');
  for (const letter of ALPHA) {
    const btn = document.createElement('button');
    btn.className   = 'learn-key';
    btn.textContent = letter;
    btn.addEventListener('click', () => showLearnDetail(letter, btn));
    container.appendChild(btn);
  }
}

function showLearnDetail(letter, btn) {
  document.querySelectorAll('.learn-key').forEach(k => k.classList.remove('active'));
  btn.classList.add('active');

  const isDynamic = DYNAMIC_LETTERS.has(letter);
  const detail    = document.getElementById('learnDetail');
  detail.innerHTML = `
    <div class="learn-letter-info">
      <div class="learn-letter-big">${letter}</div>
      <span class="learn-badge ${isDynamic ? 'dynamic' : 'static'}">
        ${isDynamic ? '⚡ Dynamic — involves motion' : '✋ Static — hold position'}
      </span>
      <p class="learn-desc">${LEARN_DESCRIPTIONS[letter] || 'No description available.'}</p>
    </div>
  `;
}

// ── Navigation ────────────────────────────────────────────────────────────────
document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const page = btn.dataset.page;
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(`page-${page}`).classList.add('active');
  });
});

// ── Accessibility controls ────────────────────────────────────────────────────
document.getElementById('themeSelect').addEventListener('change', e => {
  document.documentElement.setAttribute('data-theme',
    e.target.value === 'default' ? '' : e.target.value
  );
});
document.getElementById('fontSlider').addEventListener('input', e => {
  document.documentElement.style.setProperty('--font-size', e.target.value + 'px');
  document.getElementById('fontVal').textContent = e.target.value + 'px';
});
document.getElementById('mirrorToggle').addEventListener('change', e => {
  state.mirror = e.target.checked;
});

// ── Init ──────────────────────────────────────────────────────────────────────
buildKeyboard();
buildLearn();
renderSentence();
document.getElementById('cameraPanel').style.display = 'none';
