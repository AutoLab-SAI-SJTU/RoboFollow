(() => {
  'use strict';
  const config = window.ROBOFOLLOW_CONFIG || {};
  document.querySelectorAll('[data-link]').forEach(link => {
    if (config[link.dataset.link]) link.href = config[link.dataset.link];
  });

  const intentButtons = [...document.querySelectorAll('[data-intent]')];
  intentButtons.forEach(button => button.addEventListener('click', () => {
    intentButtons.forEach(item => {
      item.classList.toggle('active', item === button);
      item.setAttribute('aria-pressed', String(item === button));
    });
    const behind = button.dataset.intent === 'behind';
    const marker = document.getElementById('target-highlight');
    marker.style.left = behind ? '48%' : '67%';
    marker.style.top = behind ? '49%' : '74%';
    marker.style.width = behind ? '12%' : '15%';
    marker.style.height = behind ? '16%' : '20%';
    document.getElementById('intent-caption').textContent = behind
      ? 'Instruction B selects the upper-middle yellow block. The visual observation is unchanged.'
      : 'Instruction A selects the lower-right yellow block. Illustrative target highlighting on a scene from the paper.';
  }));

  const tabs = [...document.querySelectorAll('.scene-tab')];
  function selectTab(tab, focus = false) {
    tabs.forEach(item => {
      const selected = item === tab;
      item.classList.toggle('active', selected);
      item.setAttribute('aria-selected', String(selected));
      item.tabIndex = selected ? 0 : -1;
      document.getElementById(item.getAttribute('aria-controls')).hidden = !selected;
    });
    if (focus) tab.focus();
  }
  tabs.forEach((tab, i) => {
    tab.addEventListener('click', () => selectTab(tab));
    tab.addEventListener('keydown', event => {
      let next;
      if (event.key === 'ArrowRight') next = (i + 1) % tabs.length;
      if (event.key === 'ArrowLeft') next = (i - 1 + tabs.length) % tabs.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = tabs.length - 1;
      if (next !== undefined) {
        event.preventDefault();
        selectTab(tabs[next], true);
      }
    });
  });

  const data = window.ROBOFOLLOW_RESULTS;
  const fmt = value => ((Math.round((Math.abs(value) + 1e-8) * 10) / 10) * (value < 0 ? -1 : 1)).toFixed(1);
  const metricNames = { IS: 'Intent Score', ES: 'Execution Score', CR: 'Completion Rate' };
  const sceneNames = { S1: 'Scene 1 · Spatial relations', S2: 'Scene 2 · Object attributes', S3: 'Scene 3 · Motion constraints', S4: 'Scene 4 · Logical grounding' };
  const metricSelect = document.getElementById('metric-select');
  const sceneSelect = document.getElementById('scene-select');
  const modelSelect = document.getElementById('model-select');
  const escape = text => String(text).replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
  function renderResults() {
    if (!data) return;
    const metric = metricSelect.value;
    const scene = sceneSelect.value;
    const selected = data.models.find(model => model.id === modelSelect.value);
    const scores = model => (scene === 'mean' ? model.mean : model.scenes[scene]).map(value => value[metric]);
    const values = scores(selected);
    const xs = [75, 300, 525, 750];
    const y = value => 330 - value * 2.85;
    const context = scene === 'mean' ? 'Unweighted mean over four scenes' : sceneNames[scene];
    const description = `${selected.name}, ${metricNames[metric]}, ${context}. ${values.map((value, i) => `L${i}: ${fmt(value)} percent`).join('; ')}. Full values for all models follow in the table.`;
    let svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 400" role="img" aria-labelledby="chart-title chart-description"><title id="chart-title">${escape(metricNames[metric])} across L0–L3</title><desc id="chart-description">${escape(description)}</desc><g font-family="Arial, sans-serif"><rect x="48" y="28" width="81" height="302" rx="5" fill="#eceee5"/>`;
    for (const tick of [0, 25, 50, 75, 100]) {
      svg += `<line x1="55" y1="${y(tick)}" x2="768" y2="${y(tick)}" stroke="#d9dcd5" stroke-width="1"/><text x="39" y="${y(tick) + 4}" text-anchor="end" font-size="12" fill="#58656c">${tick}</text>`;
    }
    svg += '<text x="12" y="17" font-size="11" fill="#58656c">Score (%)</text>';
    const drawModel = (model, type) => {
      const vals = scores(model);
      const color = type === 'selected' ? '#c44a2c' : type === 'control' ? '#267975' : '#a9b6bb';
      let line = `<polyline points="${vals.map((value, i) => `${xs[i]},${y(value)}`).join(' ')}" fill="none" stroke="${color}" stroke-width="${type === 'selected' ? 3.6 : type === 'control' ? 2 : 1.3}" ${type === 'control' ? 'stroke-dasharray="6 6"' : ''} opacity="${type === 'other' ? '.5' : '1'}" stroke-linecap="round" stroke-linejoin="round"/>`;
      if (type === 'selected') vals.forEach((value, i) => {
        line += `<circle cx="${xs[i]}" cy="${y(value)}" r="5.5" fill="${color}" stroke="#f7f5ef" stroke-width="2"/><text x="${xs[i]}" y="${y(value) - 15}" font-size="16" font-weight="bold" text-anchor="middle" fill="${color}" stroke="#f7f5ef" stroke-width="4" paint-order="stroke">${fmt(value)}</text>`;
      });
      return line;
    };
    data.models.filter(model => model.id !== selected.id && model.family !== 'Control').forEach(model => svg += drawModel(model, 'other'));
    svg += drawModel(data.models.find(model => model.family === 'Control'), 'control');
    svg += drawModel(selected, 'selected');
    ['In distribution', 'Visual', 'Semantic', 'Visual + semantic'].forEach((label, i) => {
      svg += `<text x="${xs[i]}" y="357" font-size="15" font-weight="bold" text-anchor="middle" fill="#173247">L${i}</text><text x="${xs[i]}" y="379" font-size="11" text-anchor="middle" fill="#58656c">${label}</text>`;
    });
    svg += '</g></svg>';
    document.getElementById('results-chart').innerHTML = svg;
    document.getElementById('result-model').textContent = `${selected.name} · ${metricNames[metric].toUpperCase()}`;
    document.getElementById('score-l0').innerHTML = `${fmt(values[0])}<span>%</span>`;
    document.getElementById('score-l3').innerHTML = `${fmt(values[3])}<span>%</span>`;
    const drop = values[0] - values[3];
    document.getElementById('score-drop').innerHTML = `<strong>${fmt(Math.abs(drop))} percentage points</strong> ${drop >= 0 ? 'lower' : 'higher'} at L3.`;
    document.getElementById('score-context').textContent = scene === 'mean'
      ? 'Unweighted mean over four scenes, calculated from the rounded values in Table 1.'
      : `${sceneNames[scene]}. Values reported in Table 1.`;
    document.querySelector('#results-table caption').textContent = `${metricNames[metric]} (%) · ${context.toLowerCase()}`;
    document.getElementById('results-table-body').innerHTML = data.models.map(model => {
      const type = model.id === selected.id ? 'selected' : model.family === 'Control' ? 'control' : '';
      return `<tr class="${type}"><th scope="row">${escape(model.name)}</th><td>${model.family}</td>${scores(model).map(value => `<td>${fmt(value)}</td>`).join('')}</tr>`;
    }).join('');
  }
  [metricSelect, sceneSelect, modelSelect].forEach(select => select.addEventListener('change', renderResults));
  renderResults();

  document.getElementById('copy-citation').addEventListener('click', async () => {
    const citation = document.getElementById('bibtex');
    const status = document.getElementById('copy-status');
    const text = citation.textContent.trim();
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
      } else {
        const field = document.createElement('textarea');
        field.value = text;
        field.style.position = 'fixed';
        field.style.opacity = '0';
        document.body.append(field);
        field.select();
        const ok = document.execCommand('copy');
        field.remove();
        if (!ok) throw new Error('Copy unavailable');
      }
      status.textContent = 'BibTeX copied to clipboard.';
      document.getElementById('copy-citation').textContent = 'Copied ✓';
    } catch (_) {
      const range = document.createRange();
      range.selectNodeContents(citation);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      status.textContent = 'Citation selected. Press Ctrl+C (or ⌘C) to copy.';
    }
  });
})();
