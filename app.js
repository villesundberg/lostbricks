// LostBricks — app.js

const API_BASE = 'https://rebrickable.com/api/v3';
const STORAGE_KEY = 'lostbricks';

// ── State ──

let state = { apiKey: '', userToken: '', sets: {} };

function loadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) state = JSON.parse(raw);
  } catch (e) {
    console.error('Failed to load state', e);
  }
}

function saveState() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  // Also persist to server (fire-and-forget)
  fetch('api/state', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(state),
  }).catch(() => {});
}

async function loadStateFromServer() {
  try {
    const res = await fetch('api/state');
    const data = await res.json();
    if (data && data.apiKey) {
      state = data;
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
      return true;
    }
  } catch (e) {
    console.warn('Server state load failed, using localStorage', e);
  }
  return false;
}

// ── API ──

async function apiFetch(endpoint, { params, method = 'GET', body } = {}) {
  const url = new URL(`${API_BASE}${endpoint}`);
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined) url.searchParams.set(k, v);
    }
  }
  const opts = {
    method,
    headers: {
      'Authorization': `key ${state.apiKey}`,
      'Accept': 'application/json',
    },
  };
  if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const resp = await fetch(url, opts);
  if (!resp.ok) {
    if (resp.status === 404) return null;
    throw new Error(`API error ${resp.status}`);
  }
  if (resp.status === 204) return null;
  return resp.json();
}

function normalizeSetNum(num) {
  const s = String(num).trim();
  if (/^\d+-\d+$/.test(s)) return s;
  return s.replace(/[^\d]/g, '') + '-1';
}

async function fetchSet(setNum) {
  const data = await apiFetch(`/lego/sets/${setNum}/`, {});
  if (!data) return null;
  return {
    setNum: data.set_num,
    name: data.name,
    year: data.year,
    numParts: data.num_parts,
    imgUrl: data.set_img_url,
  };
}

async function fetchParts(setNum) {
  const parts = [];
  let page = 1;
  while (true) {
    const data = await apiFetch(`/lego/sets/${setNum}/parts/`, {
      params: { page_size: 1000, page, inc_minifig_parts: 0 },
    });
    if (!data || !data.results) break;
    for (const r of data.results) {
      if (r.is_spare) continue;
      parts.push({
        invPartId: r.inv_part_id,
        partNum: r.part.part_num,
        name: r.part.name,
        imgUrl: r.part.part_img_url,
        color: {
          id: r.color.id,
          name: r.color.name,
          rgb: r.color.rgb,
        },
        qty: r.quantity,
      });
    }
    if (!data.next) break;
    page++;
  }
  return parts;
}

// ── Screens ──

const screens = {
  apikey: document.getElementById('screen-apikey'),
  home: document.getElementById('screen-home'),
  inventory: document.getElementById('screen-inventory'),
};

let currentSet = null;

function showScreen(name) {
  for (const [k, el] of Object.entries(screens)) {
    el.classList.toggle('hidden', k !== name);
  }
}

// ── API Key Screen ──

document.getElementById('apikey-form').addEventListener('submit', (e) => {
  e.preventDefault();
  const key = document.getElementById('apikey-input').value.trim();
  if (!key) return;
  state.apiKey = key;
  saveState();
  showScreen('home');
  renderSetList();
});

// ── Settings Modal ──

document.getElementById('settings-btn').addEventListener('click', () => {
  document.getElementById('settings-apikey').value = state.apiKey;
  document.getElementById('settings-modal').classList.remove('hidden');
});

document.getElementById('settings-cancel').addEventListener('click', () => {
  document.getElementById('settings-modal').classList.add('hidden');
});

document.getElementById('settings-form').addEventListener('submit', (e) => {
  e.preventDefault();
  const key = document.getElementById('settings-apikey').value.trim();
  if (key) {
    state.apiKey = key;
    saveState();
  }
  document.getElementById('settings-modal').classList.add('hidden');
});

// ── Home Screen ──

document.getElementById('search-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const input = document.getElementById('search-input');
  const errorEl = document.getElementById('search-error');
  const raw = input.value.trim();
  if (!raw) return;

  const setNum = normalizeSetNum(raw);
  errorEl.classList.add('hidden');

  // If we already have this set, just open it
  if (state.sets[setNum]) {
    openInventory(setNum);
    return;
  }

  // Show loading
  const btn = e.target.querySelector('button');
  btn.textContent = '...';
  btn.disabled = true;

  try {
    const setInfo = await fetchSet(setNum);
    if (!setInfo) {
      errorEl.textContent = `Set ${setNum} not found.`;
      errorEl.classList.remove('hidden');
      return;
    }

    const parts = await fetchParts(setNum);
    state.sets[setNum] = {
      ...setInfo,
      parts,
      have: {},
    };
    saveState();
    renderSetList();
    openInventory(setNum);
    input.value = '';
  } catch (err) {
    errorEl.textContent = `Error: ${err.message}`;
    errorEl.classList.remove('hidden');
  } finally {
    btn.textContent = 'Load';
    btn.disabled = false;
  }
});

function getSetProgress(setData) {
  let totalQty = 0;
  let haveQty = 0;
  let completeTypes = 0;
  let checkedTypes = 0;
  for (const p of setData.parts) {
    const key = `${p.partNum}_${p.color.id}`;
    const checked = key in setData.have;
    const have = setData.have[key] || 0;
    totalQty += p.qty;
    haveQty += Math.min(have, p.qty);
    if (checked) checkedTypes++;
    if (have >= p.qty || setData.confirmed?.[key]) completeTypes++;
  }
  return { totalQty, haveQty, completeTypes, checkedTypes, totalTypes: setData.parts.length };
}

function renderSetList() {
  const container = document.getElementById('set-list');
  const setNums = Object.keys(state.sets);

  if (setNums.length === 0) {
    container.innerHTML = '<p style="color:var(--text-dim);text-align:center;padding:32px">No sets loaded yet. Enter a set number above.</p>';
    return;
  }

  container.innerHTML = setNums.map((num) => {
    const s = state.sets[num];
    const prog = getSetProgress(s);
    const pct = prog.totalQty ? Math.round((prog.haveQty / prog.totalQty) * 100) : 0;
    const isComplete = pct === 100;
    return `
      <div class="set-card" data-set="${num}">
        <img src="${s.imgUrl || ''}" alt="" loading="lazy">
        <div class="set-card-info">
          <h3>${s.name}</h3>
          <span>${num} &middot; ${s.year}</span>
        </div>
        <div class="set-card-progress ${isComplete ? '' : 'incomplete'}">
          ${pct}%<br>
          <small>${prog.haveQty}/${prog.totalQty}</small>
        </div>
      </div>
    `;
  }).join('');

  container.querySelectorAll('.set-card').forEach((card) => {
    card.addEventListener('click', () => openInventory(card.dataset.set));
  });
}

// ── Inventory Screen ──

function openInventory(setNum) {
  currentSet = setNum;
  const s = state.sets[setNum];

  document.getElementById('set-title').textContent = s.name;
  document.getElementById('set-meta').textContent = `${setNum} · ${s.year} · ${s.parts.length} unique parts`;

  renderSummary();
  renderColorBar();
  renderParts();
  showScreen('inventory');
}

document.getElementById('back-btn').addEventListener('click', () => {
  currentSet = null;
  renderSetList();
  showScreen('home');
});

document.getElementById('delete-set-btn').addEventListener('click', () => {
  if (!currentSet) return;
  if (!confirm(`Remove ${state.sets[currentSet]?.name}? Your progress will be lost.`)) return;
  delete state.sets[currentSet];
  saveState();
  currentSet = null;
  renderSetList();
  showScreen('home');
});

function renderSummary() {
  const s = state.sets[currentSet];
  const prog = getSetProgress(s);
  const pct = prog.totalQty ? Math.round((prog.haveQty / prog.totalQty) * 100) : 0;

  document.getElementById('summary-bar').innerHTML = `
    <span>${prog.haveQty} / ${prog.totalQty} pcs · ${prog.checkedTypes}/${prog.totalTypes} checked</span>
    <div class="progress-bar"><div class="progress-fill" style="width:${pct}%"></div></div>
  `;
}

function getSortedParts() {
  const s = state.sets[currentSet];
  const sort = document.getElementById('sort-select').value;
  const parts = [...s.parts];

  const checked = (p) => `${p.partNum}_${p.color.id}` in s.have ? 1 : 0;
  const deficit = (p) => {
    const key = `${p.partNum}_${p.color.id}`;
    return p.qty - Math.min(s.have[key] || 0, p.qty);
  };

  switch (sort) {
    case 'missing-first':
      parts.sort((a, b) => checked(a) - checked(b) || deficit(b) - deficit(a) || a.name.localeCompare(b.name));
      break;
    case 'complete-first':
      parts.sort((a, b) => deficit(a) - deficit(b) || a.name.localeCompare(b.name));
      break;
    case 'color':
      parts.sort((a, b) => a.color.name.localeCompare(b.color.name) || a.name.localeCompare(b.name));
      break;
    case 'name':
      parts.sort((a, b) => a.name.localeCompare(b.name));
      break;
  }

  return parts;
}

function renderColorBar() {
  const s = state.sets[currentSet];
  const colors = new Map();
  for (const p of s.parts) {
    if (!colors.has(p.color.id)) {
      colors.set(p.color.id, { name: p.color.name, rgb: p.color.rgb });
    }
  }
  const bar = document.getElementById('color-bar');
  bar.innerHTML = [...colors.entries()].map(([id, c]) =>
    `<div class="color-chip" data-color="${id}">
      <span class="color-swatch" style="background:#${c.rgb}"></span>${c.name}
    </div>`
  ).join('');
  bar.querySelectorAll('.color-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      const anchor = document.getElementById(`color-${chip.dataset.color}`);
      if (anchor) anchor.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });
}

function renderParts() {
  const s = state.sets[currentSet];
  const parts = getSortedParts();
  const hideComplete = document.getElementById('hide-complete').checked;
  const container = document.getElementById('parts-list');
  let lastColorId = null;

  container.innerHTML = parts.map((p) => {
    const key = `${p.partNum}_${p.color.id}`;
    const checked = key in s.have;
    const have = s.have[key] || 0;
    const isComplete = checked && have >= p.qty;

    const confirmed = s.confirmed?.[key];
    const done = isComplete || confirmed;

    if (hideComplete && done) return '';

    let rowClass = 'unchecked';
    if (checked) rowClass = done ? 'complete' : 'missing';

    let colorHeader = '';
    if (document.getElementById('sort-select').value === 'color' && p.color.id !== lastColorId) {
      lastColorId = p.color.id;
      colorHeader = `<div class="color-header" id="color-${p.color.id}">
        <span class="color-swatch" style="background:#${p.color.rgb}"></span>${p.color.name}
      </div>`;
    }

    return `${colorHeader}
      <div class="part-row ${rowClass}" data-key="${key}">
        <img class="part-img" src="${p.imgUrl || ''}" alt="" loading="lazy">
        <div class="part-info">
          <div class="part-name">${p.name}</div>
          <div class="part-color">
            <span class="color-swatch" style="background:#${p.color.rgb}"></span>
            ${p.color.name}
          </div>
          <div class="part-qty">${checked && have < p.qty ? `Missing ${p.qty - have} of ${p.qty}` : `Need: ${p.qty}`}</div>
        </div>
        <div class="part-controls">
          <button class="part-minus" data-key="${key}">−</button>
          <input type="number" min="0" value="${checked ? have : ''}" placeholder="?" data-key="${key}" inputmode="numeric">
          <button class="part-plus" data-key="${key}">+</button>
          <button class="part-all" data-key="${key}" data-qty="${p.qty}">All</button>
          ${checked && !isComplete ? `<button class="part-done ${confirmed ? 'is-done' : ''}" data-key="${key}">\u2713</button>` : ''}
        </div>
      </div>
    `;
  }).join('');

  // Event delegation for controls
  container.querySelectorAll('.part-minus').forEach((btn) => {
    btn.addEventListener('click', () => adjustPart(btn.dataset.key, -1));
  });
  container.querySelectorAll('.part-plus').forEach((btn) => {
    btn.addEventListener('click', () => adjustPart(btn.dataset.key, 1));
  });
  container.querySelectorAll('.part-all').forEach((btn) => {
    btn.addEventListener('click', () => setPart(btn.dataset.key, parseInt(btn.dataset.qty)));
  });
  container.querySelectorAll('.part-done').forEach((btn) => {
    btn.addEventListener('click', () => toggleConfirmed(btn.dataset.key));
  });
  container.querySelectorAll('.part-controls input').forEach((inp) => {
    inp.addEventListener('change', () => {
      const val = Math.max(0, parseInt(inp.value) || 0);
      setPart(inp.dataset.key, val);
    });
  });
}

function toggleConfirmed(key) {
  const s = state.sets[currentSet];
  if (!s.confirmed) s.confirmed = {};
  if (s.confirmed[key]) {
    delete s.confirmed[key];
  } else {
    s.confirmed[key] = true;
  }
  saveState();
  renderParts();
}

function adjustPart(key, delta) {
  const s = state.sets[currentSet];
  const current = s.have[key] || 0;
  const newVal = Math.max(0, current + delta);
  setPart(key, newVal);
}

function setPart(key, value) {
  const s = state.sets[currentSet];
  s.have[key] = value;
  if (s.confirmed) delete s.confirmed[key];
  saveState();

  // Update just this row's input and status without full re-render
  const row = document.querySelector(`.part-row[data-key="${key}"]`);
  if (row) {
    const input = row.querySelector('input');
    input.value = value;

    const part = s.parts.find((p) => `${p.partNum}_${p.color.id}` === key);
    if (part) {
      const isComplete = value >= part.qty;
      const confirmed = s.confirmed?.[key];
      const done = isComplete || confirmed;
      row.classList.remove('unchecked', 'complete', 'missing');
      row.classList.add(done ? 'complete' : 'missing');

      const qtyEl = row.querySelector('.part-qty');
      if (qtyEl) {
        qtyEl.textContent = value < part.qty ? `Missing ${part.qty - value} of ${part.qty}` : `Need: ${part.qty}`;
      }

      if (document.getElementById('hide-complete').checked && done) {
        row.style.display = 'none';
      }
    }
  }

  renderSummary();
}

// Sort / filter / bulk changes
document.getElementById('sort-select').addEventListener('change', renderParts);
document.getElementById('hide-complete').addEventListener('change', renderParts);
document.getElementById('have-all-btn').addEventListener('click', () => {
  const s = state.sets[currentSet];
  if (!s) return;
  for (const p of s.parts) {
    const key = `${p.partNum}_${p.color.id}`;
    s.have[key] = p.qty;
  }
  saveState();
  renderSummary();
  renderParts();
});

// ── Rebrickable Sync ──

async function getUserToken(username, password) {
  const body = new URLSearchParams({ username, password });
  const resp = await fetch(`${API_BASE}/users/_token/`, {
    method: 'POST',
    headers: {
      'Authorization': `key ${state.apiKey}`,
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body,
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Login failed (${resp.status}): ${text} [user=${username}, passLen=${password.length}]`);
  }
  const data = await resp.json();
  return data.user_token;
}

async function ensureSetInCollection(setNum) {
  // Try adding; ignore 4xx if already there
  try {
    await apiFetch(`/users/${state.userToken}/sets/`, {
      method: 'POST',
      body: { set_num: setNum },
    });
  } catch (e) {
    // Already exists or other non-critical error — fine
  }
}

async function fetchRemoteLostParts(setNum) {
  const all = [];
  let page = 1;
  while (true) {
    const data = await apiFetch(`/users/${state.userToken}/lost_parts/`, {
      params: { page_size: 1000, page },
    });
    if (!data || !data.results) break;
    all.push(...data.results);
    if (!data.next) break;
    page++;
  }
  // Filter to this set
  return all.filter((lp) => lp.inv_part.set_num === setNum);
}

// Pending sync plan, set by preview, consumed by confirm
let syncPlan = null;

function computeLocalLost(s) {
  const lost = new Map(); // key -> { invPartId, name, colorName, qty }
  for (const p of s.parts) {
    const key = `${p.partNum}_${p.color.id}`;
    if (!(key in s.have)) continue;
    const have = s.have[key];
    const lostQty = p.qty - Math.min(have, p.qty);
    if (lostQty > 0 && p.invPartId) {
      lost.set(key, { invPartId: p.invPartId, name: p.name, colorName: p.color.name, qty: lostQty });
    }
  }
  return lost;
}

async function syncPreview() {
  const btn = document.getElementById('sync-btn');
  const s = state.sets[currentSet];
  if (!s) return;

  if (!state.userToken) {
    document.getElementById('login-modal').classList.remove('hidden');
    return;
  }

  btn.textContent = 'Loading...';
  btn.disabled = true;

  try {
    // Re-fetch parts if we're missing invPartId
    if (s.parts.length > 0 && !s.parts[0].invPartId) {
      const fresh = await fetchParts(currentSet);
      const lookup = new Map(fresh.map((p) => [`${p.partNum}_${p.color.id}`, p.invPartId]));
      for (const p of s.parts) {
        p.invPartId = lookup.get(`${p.partNum}_${p.color.id}`);
      }
      saveState();
    }

    // Fetch remote state
    const remoteLost = await fetchRemoteLostParts(currentSet);
    const remoteMap = new Map(); // key -> { lostPartId, qty, name, colorName }
    for (const lp of remoteLost) {
      const ip = lp.inv_part;
      const key = `${ip.part.part_num}_${ip.color.id}`;
      const existing = remoteMap.get(key);
      // Rebrickable can have multiple entries per part — sum them
      if (existing) {
        existing.qty += lp.lost_quantity;
        existing.lostPartIds.push(lp.lost_part_id);
      } else {
        remoteMap.set(key, {
          lostPartIds: [lp.lost_part_id],
          qty: lp.lost_quantity,
          name: ip.part.name,
          colorName: ip.color.name,
        });
      }
    }

    const localLost = computeLocalLost(s);

    // Build diff
    const diff = [];
    const allKeys = new Set([...remoteMap.keys(), ...localLost.keys()]);
    for (const key of allKeys) {
      const remote = remoteMap.get(key);
      const local = localLost.get(key);
      const name = local?.name || remote?.name || key;
      const color = local?.colorName || remote?.colorName || '';

      if (!remote && local) {
        diff.push({ type: 'add', name, color, qty: local.qty });
      } else if (remote && !local) {
        diff.push({ type: 'remove', name, color, qty: remote.qty });
      } else if (remote.qty !== local.qty) {
        diff.push({ type: 'change', name, color, from: remote.qty, to: local.qty });
      } else {
        diff.push({ type: 'keep', name, color, qty: remote.qty });
      }
    }

    // Sort: changes first, then adds, removes, keeps
    const order = { change: 0, add: 1, remove: 2, keep: 3 };
    diff.sort((a, b) => order[a.type] - order[b.type]);

    // Store plan for confirm
    syncPlan = { remoteLost, localLost, setNum: currentSet };

    // Render preview
    const adds = diff.filter((d) => d.type === 'add').length;
    const removes = diff.filter((d) => d.type === 'remove').length;
    const changes = diff.filter((d) => d.type === 'change').length;
    const keeps = diff.filter((d) => d.type === 'keep').length;

    document.getElementById('sync-summary').textContent =
      `${adds} to add, ${removes} to remove, ${changes} to change, ${keeps} unchanged`;

    const tags = { add: '+', remove: '\u2212', change: '\u0394', keep: '=' };
    document.getElementById('sync-diff').innerHTML = diff.map((d) => {
      let qtyText;
      if (d.type === 'change') qtyText = `${d.from} \u2192 ${d.to}`;
      else qtyText = `\u00d7${d.qty}`;

      return `<div class="diff-row ${d.type}">
        <span class="diff-tag">${tags[d.type]}</span>
        <span class="diff-name">${d.name} <small style="opacity:0.7">${d.color}</small></span>
        <span class="diff-qty">${qtyText}</span>
      </div>`;
    }).join('');

    document.getElementById('sync-modal').classList.remove('hidden');
  } catch (err) {
    alert(`Failed to load preview: ${err.message}`);
  } finally {
    btn.textContent = 'Sync';
    btn.disabled = false;
  }
}

async function syncExecute() {
  if (!syncPlan) return;
  const { remoteLost, localLost, setNum } = syncPlan;
  const confirmBtn = document.getElementById('sync-confirm');
  confirmBtn.textContent = 'Syncing...';
  confirmBtn.disabled = true;

  try {
    await ensureSetInCollection(setNum);

    // Delete all existing remote lost parts for this set
    for (const lp of remoteLost) {
      await apiFetch(`/users/${state.userToken}/lost_parts/${lp.lost_part_id}/`, {
        method: 'DELETE',
      });
    }

    // Post local lost parts
    for (const [, item] of localLost) {
      await apiFetch(`/users/${state.userToken}/lost_parts/`, {
        method: 'POST',
        body: { inv_part_id: item.invPartId, lost_quantity: item.qty },
      });
    }

    syncPlan = null;
    document.getElementById('sync-modal').classList.add('hidden');

    const btn = document.getElementById('sync-btn');
    btn.textContent = 'Synced!';
    setTimeout(() => { btn.textContent = 'Sync'; }, 2000);
  } catch (err) {
    alert(`Sync failed: ${err.message}`);
  } finally {
    confirmBtn.textContent = 'Confirm Sync';
    confirmBtn.disabled = false;
  }
}

document.getElementById('sync-btn').addEventListener('click', syncPreview);
document.getElementById('sync-confirm').addEventListener('click', syncExecute);
document.getElementById('sync-cancel').addEventListener('click', () => {
  syncPlan = null;
  document.getElementById('sync-modal').classList.add('hidden');
});

// ── Login Modal ──

document.getElementById('login-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const user = document.getElementById('login-user').value.trim();
  const pass = document.getElementById('login-pass').value;
  const errorEl = document.getElementById('login-error');
  errorEl.classList.add('hidden');

  if (!user || !pass) return;

  try {
    state.userToken = await getUserToken(user, pass);
    saveState();
    document.getElementById('login-modal').classList.add('hidden');
    document.getElementById('login-pass').value = '';
    // Continue to preview
    syncPreview();
  } catch (err) {
    errorEl.textContent = err.message;
    errorEl.classList.remove('hidden');
  }
});

document.getElementById('login-cancel').addEventListener('click', () => {
  document.getElementById('login-modal').classList.add('hidden');
});

document.getElementById('login-show-pass').addEventListener('click', () => {
  const inp = document.getElementById('login-pass');
  inp.type = inp.type === 'password' ? 'text' : 'password';
});

// ── Scan integration ──

window.lostbricks = {
  getCurrentSetParts() {
    if (!currentSet || !state.sets[currentSet]) return [];
    return state.sets[currentSet].parts;
  },
  getHave(key) {
    if (!currentSet || !state.sets[currentSet]) return 0;
    return state.sets[currentSet].have[key] || 0;
  },
  getCurrentSetNum() {
    return currentSet;
  },
  incrementPart(key) {
    const s = state.sets[currentSet];
    if (!s) return;
    if (!(key in s.have)) s.have[key] = 0;
    s.have[key]++;
    saveState();
    renderSummary();
    renderParts();
  },
};

// ── Init ──

(async () => {
  await loadStateFromServer();
  if (!state.apiKey) loadState(); // fallback to localStorage
  if (!state.apiKey) {
    showScreen('apikey');
  } else {
    showScreen('home');
    renderSetList();
  }
})();
