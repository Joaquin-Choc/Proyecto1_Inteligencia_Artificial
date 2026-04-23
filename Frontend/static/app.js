const ticketForm = document.getElementById('ticketForm');
const ticketIdInput = document.getElementById('ticketId');
const subjectInput = document.getElementById('subject');
const descriptionInput = document.getElementById('description');
const trueCategoryInput = document.getElementById('trueCategory');
const examplesFileInput = document.getElementById('examplesFile');
const predictionEl = document.getElementById('prediction');
const resultTicketEl = document.getElementById('resultTicket');
const confidenceChip = document.getElementById('confidenceChip');
const probabilityBars = document.getElementById('probabilityBars');
const tokensEl = document.getElementById('tokens');
const modelNotice = document.getElementById('modelNotice');
const saveNotice = document.getElementById('saveNotice');
const uploadNotice = document.getElementById('uploadNotice');
const trainNotice = document.getElementById('trainNotice');
const trainProgressWrap = document.getElementById('trainProgressWrap');
const trainProgressBar = document.getElementById('trainProgressBar');
const trainProgressLabel = document.getElementById('trainProgressLabel');
const trainingStatusBadge = document.getElementById('trainingStatusBadge');
const trainingTotalInstances = document.getElementById('trainingTotalInstances');
const trainingAccuracy = document.getElementById('trainingAccuracy');
const trainingMacroF1 = document.getElementById('trainingMacroF1');
const trainingClasses = document.getElementById('trainingClasses');
const trainingMessage = document.getElementById('trainingMessage');
const regenTicketBtn = document.getElementById('regenTicket');
const sampleBtn = document.getElementById('sampleBtn');
const uploadExamplesBtn = document.getElementById('uploadExamplesBtn');
const retrainBtn = document.getElementById('retrainBtn');
const themeToggle = document.getElementById('themeToggle');
const themeToggleLabel = document.getElementById('themeToggleLabel');

const chartCtx = document.getElementById('probabilityChart');
let probabilityChart = null;

const sampleTickets = [
  {
    subject: 'Refund not received after cancellation',
    description: 'I cancelled my order last week, but I still have not received the refund in my bank account.'
  },
  {
    subject: 'Tracking number is not updating',
    description: 'The shipment appears stuck in transit and the tracking page has not changed for four days.'
  },
  {
    subject: 'Need help changing account email',
    description: 'I cannot access my account settings and need to update the email address linked to my profile.'
  }
];

function generateTicketId() {
  const year = new Date().getFullYear();
  const random = Math.floor(100000 + Math.random() * 900000);
  return `TKT-${year}-${random}`;
}

function formatPercent(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function setNotice(message, variant = 'info') {
  modelNotice.className = `alert alert-${variant} mt-3`;
  modelNotice.textContent = message;
  modelNotice.classList.remove('d-none');
}

function setSaveNotice(message, variant = 'success') {
  saveNotice.className = `alert alert-${variant} mt-3`;
  saveNotice.textContent = message;
  saveNotice.classList.remove('d-none');
}

function clearNotice() {
  modelNotice.classList.add('d-none');
  modelNotice.textContent = '';
}

function clearSaveNotice() {
  saveNotice.classList.add('d-none');
  saveNotice.textContent = '';
}

function setUploadNotice(message, variant = 'info') {
  uploadNotice.className = `alert alert-${variant} mt-3`;
  uploadNotice.textContent = message;
  uploadNotice.classList.remove('d-none');
}

function clearUploadNotice() {
  uploadNotice.classList.add('d-none');
  uploadNotice.textContent = '';
}

function setTrainNotice(message, variant = 'warning') {
  trainNotice.className = `alert alert-${variant} mt-3`;
  trainNotice.textContent = message;
  trainNotice.classList.remove('d-none');
}

function clearTrainNotice() {
  trainNotice.classList.add('d-none');
  trainNotice.textContent = '';
}

function setTrainProgress(progress, message) {
  if (trainProgressWrap) {
    trainProgressWrap.classList.remove('d-none');
  }
  if (trainProgressBar) {
    trainProgressBar.style.width = `${progress}%`;
  }
  if (trainProgressLabel) {
    trainProgressLabel.textContent = `${progress}%`;
  }
  if (message) {
    setTrainNotice(message, 'warning');
  }
}

function clearTrainProgress() {
  if (trainProgressWrap) {
    trainProgressWrap.classList.add('d-none');
  }
  if (trainProgressBar) {
    trainProgressBar.style.width = '0%';
  }
  if (trainProgressLabel) {
    trainProgressLabel.textContent = '0%';
  }
}

function renderTrainingMetrics(status) {
  const result = status?.result || null;
  const currentStatus = status?.status || 'idle';

  if (trainingStatusBadge) {
    const label = currentStatus === 'completed' ? 'Completado' : currentStatus === 'running' ? 'En curso' : currentStatus === 'failed' ? 'Falló' : 'Sin ejecutar';
    trainingStatusBadge.textContent = label;
    trainingStatusBadge.className = `training-card-badge ${currentStatus}`;
  }

  if (trainingTotalInstances) {
    trainingTotalInstances.textContent = result?.total_instances ?? '-';
  }

  if (trainingAccuracy) {
    trainingAccuracy.textContent = result?.accuracy != null ? Number(result.accuracy).toFixed(4) : '-';
  }

  if (trainingMacroF1) {
    trainingMacroF1.textContent = result?.macro_f1 != null ? Number(result.macro_f1).toFixed(4) : '-';
  }

  if (trainingClasses) {
    trainingClasses.textContent = Array.isArray(result?.classes) ? result.classes.length : '-';
  }

  if (trainingMessage) {
    if (currentStatus === 'running') {
      trainingMessage.textContent = status?.message || 'Entrenando...';
    } else if (currentStatus === 'completed') {
      trainingMessage.textContent = `Modelo guardado en ${result?.model_path || 'la ruta configurada'}.`;
    } else if (currentStatus === 'failed') {
      trainingMessage.textContent = status?.error || 'El último entrenamiento falló.';
    } else {
      trainingMessage.textContent = 'Aún no hay un entrenamiento registrado.';
    }
  }
}

function getSavedTheme() {
  return localStorage.getItem('theme') || 'dark';
}

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('theme', theme);
  if (themeToggleLabel) {
    themeToggleLabel.textContent = theme === 'dark' ? 'Dark mode' : 'Light mode';
  }
}

function toggleTheme() {
  applyTheme(getSavedTheme() === 'dark' ? 'light' : 'dark');
}

function renderProbabilities(probabilities) {
  const entries = Object.entries(probabilities || {}).sort((a, b) => b[1] - a[1]);

  probabilityBars.innerHTML = entries.length
    ? entries.map(([label, probability]) => `
      <div class="probability-item">
        <div class="probability-row">
          <strong>${label}</strong>
          <span>${formatPercent(probability)}</span>
        </div>
        <div class="bar-track"><div class="bar-fill" style="width:${Math.max(probability * 100, 2)}%"></div></div>
      </div>
    `).join('')
    : '<p class="text-muted mb-0">No hay probabilidades disponibles.</p>';

  const labels = entries.map(([label]) => label);
  const values = entries.map(([, probability]) => probability * 100);

  if (probabilityChart) {
    probabilityChart.destroy();
  }

  probabilityChart = new Chart(chartCtx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Probabilidad (%)',
        data: values,
        borderRadius: 10,
        backgroundColor: entries.map((_, index) => index === 0 ? 'rgba(15, 118, 110, 0.9)' : 'rgba(245, 158, 11, 0.7)'),
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (context) => `${context.formattedValue}%`
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          ticks: { callback: (value) => `${value}%` }
        }
      }
    }
  });
}

function renderTokens(tokens) {
  tokensEl.innerHTML = tokens && tokens.length
    ? tokens.map((token) => `<span class="token-pill">${token}</span>`).join('')
    : '<span class="text-muted">-</span>';
}

function setLoading(isLoading) {
  const submitButton = ticketForm.querySelector('button[type="submit"]');
  submitButton.disabled = isLoading;
  submitButton.textContent = isLoading ? 'Procesando...' : 'Clasificar y guardar si aplica';
}

async function submitFeedback({ ticketId, subject, description, predictedCategory, trueCategory }) {
  const feedbackResponse = await fetch('/api/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      ticket_id: ticketId,
      subject,
      description,
      predicted_category: predictedCategory,
      true_category: trueCategory,
    }),
  });

  const feedbackData = await feedbackResponse.json();
  if (!feedbackResponse.ok) {
    throw new Error(feedbackData.error || 'No se pudo guardar el feedback.');
  }

  return feedbackData;
}

async function uploadExamplesFile() {
  const selectedFile = examplesFileInput?.files?.[0];
  if (!selectedFile) {
    setUploadNotice('Selecciona un archivo CSV antes de cargarlo.', 'warning');
    return;
  }

  const formData = new FormData();
  formData.append('file', selectedFile);

  uploadExamplesBtn.disabled = true;
  uploadExamplesBtn.textContent = 'Cargando...';
  clearUploadNotice();

  try {
    const response = await fetch('/api/upload-examples', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'No se pudo cargar el archivo.');
    }

    setUploadNotice(
      `Se cargaron ${data.imported_count} ejemplos desde ${data.filename}. Ya quedaron disponibles para reentrenamiento.`,
      'success'
    );
    if (examplesFileInput) {
      examplesFileInput.value = '';
    }
  } catch (error) {
    setUploadNotice(error.message, 'danger');
  } finally {
    uploadExamplesBtn.disabled = false;
    uploadExamplesBtn.textContent = 'Cargar archivo a la IA';
  }
}

async function getTrainingStatus() {
  const response = await fetch('/api/train/status');
  return response.json();
}

async function waitForTrainingCompletion(onStatusUpdate) {
  for (;;) {
    const status = await getTrainingStatus();

    if (typeof onStatusUpdate === 'function') {
      onStatusUpdate(status);
    }

    if (status.status === 'completed') {
      return status;
    }

    if (status.status === 'failed') {
      throw new Error(status.error || 'El reentrenamiento falló.');
    }

    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
}

async function retrainModel() {
  retrainBtn.disabled = true;
  retrainBtn.textContent = 'Reentrenando...';
  clearTrainNotice();
  setTrainProgress(1, 'Iniciando reentrenamiento...');

  try {
    const response = await fetch('/api/train/retrain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || 'No se pudo iniciar el reentrenamiento.');
    }

    setTrainNotice('Reentrenamiento iniciado. Espera a que termine...', 'warning');
    const finalStatus = await waitForTrainingCompletion((status) => {
      const progress = Number.isFinite(status?.progress) ? status.progress : 1;
      setTrainProgress(progress, status?.message || 'Entrenando...');
      renderTrainingMetrics(status);
    });
    setTrainProgress(finalStatus.progress ?? 100, finalStatus.message || 'Entrenamiento finalizado.');
    renderTrainingMetrics(finalStatus);

    const accuracy = finalStatus.result?.accuracy;
    const macroF1 = finalStatus.result?.macro_f1;
    const totalInstances = finalStatus.result?.total_instances;

    setTrainNotice(
      `Entrenamiento completado. Instancias: ${totalInstances ?? 'N/A'}, Accuracy: ${accuracy?.toFixed?.(4) ?? 'N/A'}, Macro F1: ${macroF1?.toFixed?.(4) ?? 'N/A'}.`,
      'success'
    );
    setTrainProgress(100, 'Entrenamiento finalizado.');
  } catch (error) {
    setTrainNotice(error.message, 'danger');
  } finally {
    retrainBtn.disabled = false;
    retrainBtn.textContent = 'Reentrenar IA';
  }
}

async function submitTicket(event) {
  event.preventDefault();

  if (!window.APP_CONFIG.modelLoaded) {
    setNotice('El modelo aún no está cargado. Ejecuta primero el entrenamiento.', 'warning');
    return;
  }

  const payload = {
    ticket_id: ticketIdInput.value.trim(),
    subject: subjectInput.value.trim(),
    description: descriptionInput.value.trim(),
    true_category: trueCategoryInput.value.trim(),
  };

  if (!payload.subject && !payload.description) {
    setNotice('Debes completar subject o description para clasificar.', 'warning');
    return;
  }

  setLoading(true);
  clearNotice();
  clearSaveNotice();

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'No se pudo clasificar el ticket.');
    }

    predictionEl.textContent = data.prediction || '-';
    resultTicketEl.textContent = data.ticket_id || payload.ticket_id;
    confidenceChip.textContent = data.prediction ? `${(data.confidence * 100).toFixed(1)}% de confianza` : 'Sin resultado';
    renderProbabilities(data.probabilities);
    renderTokens(data.tokens);

    if (payload.true_category) {
      if (!data.prediction) {
        throw new Error('No hay predicción válida para asociar al feedback.');
      }

      await submitFeedback({
        ticketId: data.ticket_id || payload.ticket_id,
        subject: payload.subject,
        description: payload.description,
        predictedCategory: data.prediction,
        trueCategory: payload.true_category,
      });

      setSaveNotice(`Feedback guardado en base de datos como ${payload.true_category}.`);
    }
  } catch (error) {
    setNotice(error.message, 'danger');
  } finally {
    setLoading(false);
  }
}

function fillSample() {
  const sample = sampleTickets[Math.floor(Math.random() * sampleTickets.length)];
  ticketIdInput.value = generateTicketId();
  subjectInput.value = sample.subject;
  descriptionInput.value = sample.description;
}

regenTicketBtn?.addEventListener('click', () => {
  ticketIdInput.value = generateTicketId();
});

sampleBtn?.addEventListener('click', fillSample);
uploadExamplesBtn?.addEventListener('click', uploadExamplesFile);
retrainBtn?.addEventListener('click', retrainModel);
ticketForm?.addEventListener('submit', submitTicket);
themeToggle?.addEventListener('click', toggleTheme);

ticketIdInput.value = ticketIdInput.value || generateTicketId();

applyTheme(getSavedTheme());
clearTrainProgress();

getTrainingStatus()
  .then((status) => {
    renderTrainingMetrics(status);
    if (status?.status === 'running') {
      setTrainProgress(status.progress ?? 1, status.message || 'Entrenamiento en curso...');
    }
  })
  .catch(() => {
    renderTrainingMetrics({ status: 'idle', result: null });
  });

if (!window.APP_CONFIG.modelLoaded) {
  setNotice('El frontend está listo, pero el modelo no se ha cargado todavía.', 'warning');
}