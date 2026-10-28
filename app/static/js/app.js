// State management
let currentPage = 1;
const pageSize = 12;

document.addEventListener('DOMContentLoaded', () => {
    // Load first page of passengers
    loadPassengers(1);

    // Initial prediction with default form values
    triggerInitialPrediction();
});

function loadPreset(preset) {
    document.getElementById('nameInput').value = preset.name;
    document.getElementById('pclassInput').value = preset.pclass;
    
    if (preset.sex === 'female') {
        document.getElementById('sexFemale').checked = true;
    } else {
        document.getElementById('sexMale').checked = true;
    }

    document.getElementById('ageRange').value = preset.age;
    document.getElementById('ageVal').innerText = preset.age;
    document.getElementById('ageInput').value = preset.age;

    document.getElementById('sibspInput').value = preset.sibsp;
    document.getElementById('parchInput').value = preset.parch;
    document.getElementById('ticketInput').value = preset.ticket;
    document.getElementById('fareInput').value = preset.fare;

    // Auto-trigger prediction
    document.getElementById('predictionForm').requestSubmit();
}

function resetForm() {
    document.getElementById('predictionForm').reset();
    document.getElementById('ageVal').innerText = '34.5';
    document.getElementById('ageRange').value = '34.5';
    triggerInitialPrediction();
}

function triggerInitialPrediction() {
    const form = document.getElementById('predictionForm');
    if (form) {
        form.requestSubmit();
    }
}

async function handlePredict(event) {
    event.preventDefault();
    const btn = document.getElementById('predictBtn');
    const originalText = btn.innerHTML;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin me-2"></i> Computing...';
    btn.disabled = true;

    const payload = {
        Name: document.getElementById('nameInput').value,
        Pclass: parseInt(document.getElementById('pclassInput').value, 10),
        Sex: document.querySelector('input[name="Sex"]:checked').value,
        Age: parseFloat(document.getElementById('ageRange').value),
        SibSp: parseInt(document.getElementById('sibspInput').value, 10),
        Parch: parseInt(document.getElementById('parchInput').value, 10),
        Ticket: document.getElementById('ticketInput').value,
        Fare: parseFloat(document.getElementById('fareInput').value)
    };

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const resData = await response.json();
        if (resData.status === 'success') {
            updatePredictionUI(resData.data);
        } else {
            alert('Prediction error: ' + (resData.message || 'Unknown error'));
        }
    } catch (err) {
        console.error('API Error:', err);
        alert('Network or server error during prediction.');
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
}

function updatePredictionUI(data) {
    const isSurvived = data.survived === 1;
    const probPct = data.probability_percentage;

    // Title & Confidence
    const statusTitle = document.getElementById('statusTitle');
    const statusSub = document.getElementById('statusSub');
    const iconWrapper = document.getElementById('statusIconWrapper');
    const icon = document.getElementById('statusIcon');

    if (isSurvived) {
        statusTitle.innerText = 'Survived';
        statusTitle.className = 'fw-bold mb-1 text-success';
        statusSub.innerText = data.confidence;
        iconWrapper.className = 'status-icon-wrapper survived mb-3';
        icon.className = 'fa-solid fa-life-ring text-success display-4';
    } else {
        statusTitle.innerText = 'Did Not Survive';
        statusTitle.className = 'fw-bold mb-1 text-danger';
        statusSub.innerText = data.confidence;
        iconWrapper.className = 'status-icon-wrapper perished mb-3';
        icon.className = 'fa-solid fa-skull text-danger display-4';
    }

    // Gauge
    const probPercent = document.getElementById('probPercent');
    const probBar = document.getElementById('probBar');
    probPercent.innerText = `${probPct.toFixed(2)}%`;
    probBar.style.width = `${Math.max(1, probPct)}%`;

    if (probPct >= 50) {
        probBar.className = 'progress-bar progress-bar-striped progress-bar-animated bg-success';
    } else if (probPct >= 30) {
        probBar.className = 'progress-bar progress-bar-striped progress-bar-animated bg-warning';
    } else {
        probBar.className = 'progress-bar progress-bar-striped progress-bar-animated bg-danger';
    }

    // Insights List
    const insightsList = document.getElementById('insightsList');
    insightsList.innerHTML = '';
    data.insights.forEach(item => {
        const li = document.createElement('li');
        const borderColor = isSurvived ? 'border-success' : 'border-danger';
        li.className = `p-2 rounded bg-light border-start border-3 ${borderColor} small`;
        li.innerText = item;
        insightsList.appendChild(li);
    });
}

// Dataset Explorer functions
async function loadPassengers(page = 1) {
    currentPage = page;
    const tbody = document.getElementById('passengersBody');
    tbody.innerHTML = `<tr><td colspan="10" class="text-center py-4 text-muted"><div class="spinner-border spinner-border-sm text-primary me-2"></div> Loading passengers...</td></tr>`;

    const search = document.getElementById('searchFilter').value;
    const pclass = document.getElementById('classFilter').value;
    const sex = document.getElementById('sexFilter').value;
    const survived = document.getElementById('survivalFilter').value;

    const params = new URLSearchParams({
        page: page,
        page_size: pageSize,
        search: search,
        pclass: pclass,
        sex: sex,
        survived: survived
    });

    try {
        const res = await fetch(`/api/passengers?${params.toString()}`);
        const data = await res.json();
        renderPassengersTable(data);
    } catch (err) {
        tbody.innerHTML = `<tr><td colspan="10" class="text-center py-4 text-danger">Failed to load passengers data.</td></tr>`;
    }
}

function renderPassengersTable(data) {
    const tbody = document.getElementById('passengersBody');
    tbody.innerHTML = '';

    if (!data.records || data.records.length === 0) {
        tbody.innerHTML = `<tr><td colspan="10" class="text-center py-4 text-muted">No passengers match the current filters.</td></tr>`;
        return;
    }

    data.records.forEach(row => {
        const tr = document.createElement('tr');
        const isSurvived = row.Survived === 1;
        const outcomeBadge = isSurvived 
            ? '<span class="badge bg-success-subtle text-success border border-success px-2 py-1"><i class="fa-solid fa-check me-1"></i> Survived</span>'
            : '<span class="badge bg-danger-subtle text-danger border border-danger px-2 py-1"><i class="fa-solid fa-xmark me-1"></i> Perished</span>';

        const classBadge = row.Pclass === 1 
            ? '<span class="badge bg-warning text-dark">1st</span>'
            : (row.Pclass === 2 ? '<span class="badge bg-info text-dark">2nd</span>' : '<span class="badge bg-secondary">3rd</span>');

        const sexIcon = row.Sex === 'female' 
            ? '<i class="fa-solid fa-venus text-danger me-1"></i> female' 
            : '<i class="fa-solid fa-mars text-primary me-1"></i> male';

        const ageDisplay = row.Age !== '' && row.Age !== null ? row.Age : '<span class="text-muted fst-italic">Missing</span>';
        const fareDisplay = row.Fare !== '' && row.Fare !== null ? `£${parseFloat(row.Fare).toFixed(2)}` : '<span class="text-muted fst-italic">Missing</span>';

        tr.innerHTML = `
            <td class="text-muted small">${row.PassengerId}</td>
            <td class="fw-semibold">${row.Name}</td>
            <td>${outcomeBadge}</td>
            <td>${classBadge}</td>
            <td class="text-capitalize small">${sexIcon}</td>
            <td>${ageDisplay}</td>
            <td>${row.SibSp} sib / ${row.Parch} parch</td>
            <td class="font-monospace small text-muted">${row.Ticket}</td>
            <td>${fareDisplay}</td>
            <td>
                <button class="btn btn-outline-primary btn-sm" onclick='loadPassengerIntoPredictor(${JSON.stringify(row)})'>
                    <i class="fa-solid fa-arrow-up-right-from-square me-1"></i> Test
                </button>
            </td>
        `;
        tbody.appendChild(tr);
    });

    // Pagination info
    const start = (data.page - 1) * data.page_size + 1;
    const end = Math.min(start + data.records.length - 1, data.total);
    document.getElementById('paginationInfo').innerText = `Showing ${start} to ${end} of ${data.total} passengers`;

    // Pagination Controls
    renderPagination(data.page, data.total_pages);
}

function renderPagination(current, totalPages) {
    const ul = document.getElementById('paginationControls');
    ul.innerHTML = '';

    if (totalPages <= 1) return;

    // Previous
    const prevLi = document.createElement('li');
    prevLi.className = `page-item ${current === 1 ? 'disabled' : ''}`;
    prevLi.innerHTML = `<button class="page-link" onclick="loadPassengers(${current - 1})">&laquo;</button>`;
    ul.appendChild(prevLi);

    // Page numbers
    const startPage = Math.max(1, current - 2);
    const endPage = Math.min(totalPages, current + 2);

    for (let p = startPage; p <= endPage; p++) {
        const li = document.createElement('li');
        li.className = `page-item ${p === current ? 'active' : ''}`;
        li.innerHTML = `<button class="page-link" onclick="loadPassengers(${p})">${p}</button>`;
        ul.appendChild(li);
    }

    // Next
    const nextLi = document.createElement('li');
    nextLi.className = `page-item ${current === totalPages ? 'disabled' : ''}`;
    nextLi.innerHTML = `<button class="page-link" onclick="loadPassengers(${current + 1})">&raquo;</button>`;
    ul.appendChild(nextLi);
}

let filterTimeout = null;
function filterPassengers() {
    clearTimeout(filterTimeout);
    filterTimeout = setTimeout(() => {
        loadPassengers(1);
    }, 250);
}

function loadPassengerIntoPredictor(row) {
    // Switch to predictor tab
    const triggerTab = document.querySelector('#predictor-tab');
    bootstrap.Tab.getInstance(triggerTab)?.show() || new bootstrap.Tab(triggerTab).show();

    // Populate inputs
    document.getElementById('nameInput').value = row.Name || '';
    document.getElementById('pclassInput').value = row.Pclass || 3;
    
    if (row.Sex === 'female') {
        document.getElementById('sexFemale').checked = true;
    } else {
        document.getElementById('sexMale').checked = true;
    }

    const age = row.Age !== '' && row.Age !== null ? row.Age : 28.0;
    document.getElementById('ageRange').value = age;
    document.getElementById('ageVal').innerText = age;
    document.getElementById('ageInput').value = age;

    document.getElementById('sibspInput').value = row.SibSp || 0;
    document.getElementById('parchInput').value = row.Parch || 0;
    document.getElementById('ticketInput').value = row.Ticket || '';
    document.getElementById('fareInput').value = row.Fare !== '' && row.Fare !== null ? row.Fare : 14.45;

    // Trigger prediction
    document.getElementById('predictionForm').requestSubmit();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// API Playground
async function testLiveApi() {
    const editor = document.getElementById('apiPayload');
    const out = document.getElementById('apiResponse');

    let body;
    try {
        body = JSON.parse(editor.value);
    } catch (e) {
        out.innerText = 'Invalid JSON: ' + e.message;
        return;
    }

    out.innerText = 'Sending request to POST /api/predict...';

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        const jsonRes = await res.json();
        out.innerText = JSON.stringify(jsonRes, null, 2);
    } catch (err) {
        out.innerText = 'Network error: ' + err.message;
    }
}
