"use strict";

(() => {
  const TRAINING_STEPS = 6000;
  const LEARNING_RATE = 0.8;
  const FRAME_INTERVAL = 25;
  const HIDDEN_UNITS = 4;
  const samples = [
    { x: 0, y: 0, target: 0 },
    { x: 0, y: 1, target: 1 },
    { x: 1, y: 0, target: 1 },
    { x: 1, y: 1, target: 0 },
  ];

  const networkCanvas = document.querySelector("#network-canvas");
  const heatmapCanvas = document.querySelector("#heatmap-canvas");
  const lossCanvas = document.querySelector("#loss-canvas");
  const context = networkCanvas.getContext("2d");
  const heatmapContext = heatmapCanvas.getContext("2d");
  const lossContext = lossCanvas.getContext("2d");
  const startButton = document.querySelector("#start-button");
  const statusLabel = document.querySelector("#model-status");
  const stepValue = document.querySelector("#step-value");
  const stepCaption = document.querySelector("#step-caption");
  const lossValue = document.querySelector("#loss-value");
  const chartLoss = document.querySelector("#chart-loss");
  const accuracyValue = document.querySelector("#accuracy-value");
  const accuracyCaption = document.querySelector("#accuracy-caption");
  const trainingRows = [...document.querySelectorAll("[data-sample]")];
  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  let model;
  let step = 0;
  let lossHistory = [];
  let animationId = null;
  let lastFrameTime = 0;
  let surfaceYaw = Math.PI / 4;
  let surfaceDragX = null;
  let networkWidth = 0;
  let networkHeight = 0;
  let heatmapSize = 0;
  let lossWidth = 0;
  let lossHeight = 0;

  function sigmoid(value) {
    const bounded = Math.max(-60, Math.min(60, value));
    return 1 / (1 + Math.exp(-bounded));
  }

  function createNetwork() {
    return {
      hiddenWeights: [
        [-0.3523344703, -0.6983016522],
        [0.3018689461, -0.8551274267],
        [0.0717640086, -0.2686221662],
        [-0.8840021505, 0.0148714664],
      ],
      hiddenBiases: [-0.9250086831, -0.1327086327, -0.8602891529, -0.8185739733],
      outputWeights: [-0.1509616217, 0.6537042493, -0.7523960777, -0.5535220708],
      outputBias: 0.2548664448,
    };
  }

  function predict(x, y) {
    const hidden = model.hiddenWeights.map((weights, index) =>
      sigmoid(weights[0] * x + weights[1] * y + model.hiddenBiases[index]),
    );
    const output = sigmoid(
      hidden.reduce((sum, activation, index) => sum + activation * model.outputWeights[index], model.outputBias),
    );
    return { output, hidden };
  }

  function trainStep() {
    for (const sample of samples) {
      const { output, hidden } = predict(sample.x, sample.y);
      const outputDelta = (sample.target - output) * output * (1 - output);
      const hiddenDeltas = hidden.map(
        (activation, index) =>
          activation * (1 - activation) * model.outputWeights[index] * outputDelta,
      );

      for (let index = 0; index < HIDDEN_UNITS; index += 1) {
        model.outputWeights[index] += LEARNING_RATE * outputDelta * hidden[index];
      }
      model.outputBias += LEARNING_RATE * outputDelta;

      for (let index = 0; index < HIDDEN_UNITS; index += 1) {
        model.hiddenWeights[index][0] += LEARNING_RATE * hiddenDeltas[index] * sample.x;
        model.hiddenWeights[index][1] += LEARNING_RATE * hiddenDeltas[index] * sample.y;
        model.hiddenBiases[index] += LEARNING_RATE * hiddenDeltas[index];
      }
    }
  }

  function getLoss() {
    return samples.reduce((sum, sample) => {
      const error = predict(sample.x, sample.y).output - sample.target;
      return sum + error * error;
    }, 0) / samples.length;
  }

  function resizeCanvas(canvas, ctx, width, height) {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const pixelWidth = Math.max(1, Math.round(width * ratio));
    const pixelHeight = Math.max(1, Math.round(height * ratio));
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
    }
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { width, height };
  }

  function measureCanvases() {
    const networkRect = networkCanvas.getBoundingClientRect();
    const heatmapRect = heatmapCanvas.getBoundingClientRect();
    const lossRect = lossCanvas.getBoundingClientRect();
    networkWidth = networkRect.width;
    networkHeight = networkRect.height;
    heatmapSize = Math.min(heatmapRect.width, heatmapRect.height);
    lossWidth = lossRect.width;
    lossHeight = lossRect.height;
    resizeCanvas(networkCanvas, context, networkWidth, networkHeight);
    resizeCanvas(heatmapCanvas, heatmapContext, heatmapSize, heatmapSize);
    resizeCanvas(lossCanvas, lossContext, lossWidth, lossHeight);
  }

  function drawNetwork(timestamp = 0) {
    const width = networkWidth;
    const height = networkHeight;
    if (!width || !height) return;
    context.clearRect(0, 0, width, height);

    const mobile = width < 440;
    const inputX = width * (mobile ? 0.13 : 0.12);
    const hiddenX = width * (mobile ? 0.54 : 0.52);
    const outputX = width * (mobile ? 0.88 : 0.86);
    const inputYs = [height * 0.35, height * 0.72];
    const hiddenYs = Array.from({ length: HIDDEN_UNITS }, (_, index) =>
      height * (0.17 + (0.66 * index) / (HIDDEN_UNITS - 1)),
    );
    const outputY = height * 0.535;
    const nodeRadius = Math.max(14, Math.min(21, width * 0.038));
    const { hidden } = predict(0.5, 0.5);

    for (let unit = 0; unit < HIDDEN_UNITS; unit += 1) {
      for (let input = 0; input < inputYs.length; input += 1) {
        drawConnection(inputX, inputYs[input], hiddenX, hiddenYs[unit], model.hiddenWeights[unit][input], unit * 0.7 + input * 0.3, timestamp);
      }
      drawConnection(hiddenX, hiddenYs[unit], outputX, outputY, model.outputWeights[unit], unit * 0.55 + 1.1, timestamp);
    }

    drawNeuron(inputX, inputYs[0], nodeRadius, "#e5ecf7", "#7185a3", "x₁", null);
    drawNeuron(inputX, inputYs[1], nodeRadius, "#e5ecf7", "#7185a3", "x₂", null);
    hiddenYs.forEach((y, index) => {
      const strength = hidden[index];
      const green = Math.round(91 + strength * 100);
      drawNeuron(hiddenX, y, nodeRadius, `rgba(32, ${green}, 125, ${0.38 + strength * 0.62})`, "#27816a", `h${index + 1}`, strength);
    });
    const output = predict(0.5, 0.5).output;
    drawNeuron(outputX, outputY, nodeRadius + 2, `rgba(115, 86, 199, ${0.32 + output * 0.3})`, "#7455d2", "ŷ", output);
  }

  function drawConnection(x1, y1, x2, y2, weight, phase, timestamp) {
    const positive = weight >= 0;
    const strength = Math.min(Math.abs(weight) / 3.2, 1);
    const color = positive ? "22, 132, 108" : "204, 76, 91";
    context.save();
    context.beginPath();
    context.moveTo(x1, y1);
    context.lineTo(x2, y2);
    context.lineWidth = 0.7 + strength * 2.4;
    context.strokeStyle = `rgba(${color}, ${0.2 + strength * 0.55})`;
    if (strength > 0.24) {
      context.shadowBlur = 7 * strength;
      context.shadowColor = `rgba(${color}, .45)`;
    }
    context.stroke();
    context.restore();

    if (statusLabel.textContent === "Training" && Math.abs(weight) > 0.18) {
      const progress = ((timestamp / 1050 + phase) % 1);
      const travel = positive ? progress : 1 - progress;
      const px = x1 + (x2 - x1) * travel;
      const py = y1 + (y2 - y1) * travel;
      context.beginPath();
      context.arc(px, py, 1.7 + strength, 0, Math.PI * 2);
      context.fillStyle = `rgba(${color}, ${0.38 + strength * 0.55})`;
      context.shadowBlur = 8;
      context.shadowColor = `rgba(${color}, .75)`;
      context.fill();
      context.shadowBlur = 0;
    }
  }

  function drawNeuron(x, y, radius, fill, stroke, label, activation) {
    const gradient = context.createRadialGradient(x - radius * 0.35, y - radius * 0.4, 1, x, y, radius * 1.5);
    gradient.addColorStop(0, fill);
    gradient.addColorStop(1, "#f2f5fa");
    context.save();
    context.beginPath();
    context.arc(x, y, radius + 5, 0, Math.PI * 2);
    context.fillStyle = "rgba(41, 58, 84, .06)";
    context.fill();
    context.beginPath();
    context.arc(x, y, radius, 0, Math.PI * 2);
    context.fillStyle = gradient;
    context.strokeStyle = stroke;
    context.lineWidth = 1.15;
    context.shadowBlur = 13;
    context.shadowColor = fill;
    context.fill();
    context.stroke();
    context.shadowBlur = 0;
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillStyle = "#26344d";
    context.font = `500 ${Math.max(10, Math.min(13, radius * 0.7))}px "DM Mono", monospace`;
    context.fillText(label, x, y);
    if (activation !== null) {
      context.fillStyle = "#68758c";
      context.font = '9px "DM Mono", monospace';
      context.fillText(activation.toFixed(2), x, y + radius + 12);
    }
    context.restore();
  }

  function drawHeatmap() {
    const size = heatmapSize;
    if (!size) return;
    heatmapContext.clearRect(0, 0, size, size);
    const segments = 20;
    const scale = size * 0.58;
    const centerX = size * 0.5;
    const centerY = size * 0.62;
    const elevation = 0.42;
    const heightScale = 0.72;
    const project = (x, y, output) => {
      const offsetX = x - 0.5;
      const offsetY = y - 0.5;
      const horizontal = offsetX * Math.cos(surfaceYaw) - offsetY * Math.sin(surfaceYaw);
      const depth = offsetX * Math.sin(surfaceYaw) + offsetY * Math.cos(surfaceYaw);
      return {
        x: centerX + horizontal * scale,
        y: centerY + depth * scale * elevation - output * scale * heightScale,
        depth,
      };
    };

    const vertices = Array.from({ length: segments + 1 }, (_, row) =>
      Array.from({ length: segments + 1 }, (_, column) => {
        const x = column / segments;
        const y = row / segments;
        const output = predict(x, y).output;
        return { ...project(x, y, output), output };
      }),
    );
    const cells = [];
    for (let row = 0; row < segments; row += 1) {
      for (let column = 0; column < segments; column += 1) {
        const corners = [
          vertices[row][column],
          vertices[row][column + 1],
          vertices[row + 1][column + 1],
          vertices[row + 1][column],
        ];
        cells.push({
          corners,
          depth: corners.reduce((sum, point) => sum + point.depth, 0) / corners.length,
        });
      }
    }
    cells.sort((first, second) => first.depth - second.depth);

    for (const cell of cells) {
      const [frontLeft, frontRight, backRight, backLeft] = cell.corners;
      const output = cell.corners.reduce((sum, point) => sum + point.output, 0) / 4;
      const slopeX = ((frontRight.output - frontLeft.output) + (backRight.output - backLeft.output)) / 2;
      const slopeY = ((backLeft.output - frontLeft.output) + (backRight.output - frontRight.output)) / 2;
      const normalLength = Math.hypot(slopeX, slopeY, 1);
      const light = Math.max(0, (-slopeX * -0.35 - slopeY * -0.5 + 0.8) / normalLength);
      const shade = 0.76 + light * 0.24;
      const color = heatColor(output).map((channel) => Math.round(channel * shade));

      heatmapContext.beginPath();
      heatmapContext.moveTo(frontLeft.x, frontLeft.y);
      for (const point of cell.corners.slice(1)) heatmapContext.lineTo(point.x, point.y);
      heatmapContext.closePath();
      heatmapContext.fillStyle = `rgb(${color.join(",")})`;
      heatmapContext.fill();
      heatmapContext.strokeStyle = "rgba(45, 59, 91, .2)";
      heatmapContext.lineWidth = 0.65;
      heatmapContext.stroke();
    }

    drawSurfaceAxes(project);
    drawSurfaceSamples(project);
  }

  function drawSurfaceAxes(project) {
    const axes = [
      { end: project(1, 0, 0), label: "INPUT 1", offsetX: 8, offsetY: 12 },
      { end: project(0, 1, 0), label: "INPUT 2", offsetX: -38, offsetY: 12 },
      { end: project(0, 0, 1), label: "OUTPUT", offsetX: -28, offsetY: -7 },
    ];
    const origin = project(0, 0, 0);
    heatmapContext.save();
    heatmapContext.strokeStyle = "#52627f";
    heatmapContext.fillStyle = "#52627f";
    heatmapContext.lineWidth = 1.15;
    heatmapContext.font = '8px "DM Mono", monospace';
    heatmapContext.textBaseline = "middle";

    for (const axis of axes) {
      heatmapContext.beginPath();
      heatmapContext.moveTo(origin.x, origin.y);
      heatmapContext.lineTo(axis.end.x, axis.end.y);
      heatmapContext.stroke();
      heatmapContext.beginPath();
      heatmapContext.arc(axis.end.x, axis.end.y, 2, 0, Math.PI * 2);
      heatmapContext.fill();
      heatmapContext.textAlign = axis.offsetX < 0 ? "right" : "left";
      heatmapContext.fillText(axis.label, axis.end.x + axis.offsetX, axis.end.y + axis.offsetY);
    }

    heatmapContext.fillStyle = "#62718b";
    heatmapContext.font = '7px "DM Mono", monospace';
    heatmapContext.textAlign = "center";
    heatmapContext.fillText("0", origin.x - 4, origin.y + 10);
    const halfInput = project(0.5, 0, 0);
    const halfOutput = project(0, 0, 0.5);
    heatmapContext.fillText("0.5", halfInput.x + 2, halfInput.y + 10);
    heatmapContext.textAlign = "right";
    heatmapContext.fillText("0.5", halfOutput.x - 7, halfOutput.y);
    heatmapContext.textAlign = "left";
    heatmapContext.fillText("1", axes[2].end.x - 2, axes[2].end.y - 7);
    heatmapContext.restore();
  }

  function drawSurfaceSamples(project) {
    for (const sample of samples) {
      const output = predict(sample.x, sample.y).output;
      const point = project(sample.x, sample.y, output);
      const baseline = project(sample.x, sample.y, 0);

      heatmapContext.beginPath();
      heatmapContext.moveTo(baseline.x, baseline.y);
      heatmapContext.lineTo(point.x, point.y);
      heatmapContext.strokeStyle = "rgba(38, 51, 76, .42)";
      heatmapContext.lineWidth = 1;
      heatmapContext.setLineDash([2, 2]);
      heatmapContext.stroke();
      heatmapContext.setLineDash([]);

      heatmapContext.beginPath();
      heatmapContext.arc(point.x, point.y, 5, 0, Math.PI * 2);
      heatmapContext.fillStyle = sample.target === 0 ? "#33415d" : "#ffffff";
      heatmapContext.fill();
      heatmapContext.lineWidth = 1.5;
      heatmapContext.strokeStyle = "#40516f";
      heatmapContext.stroke();
    }
  }

  function heatColor(value) {
    const stops = [
      [0, [48, 34, 96]],
      [0.35, [78, 111, 190]],
      [0.68, [65, 186, 168]],
      [1, [252, 218, 112]],
    ];
    let lower = stops[0];
    let upper = stops[stops.length - 1];
    for (let index = 0; index < stops.length - 1; index += 1) {
      if (value >= stops[index][0] && value <= stops[index + 1][0]) {
        lower = stops[index];
        upper = stops[index + 1];
        break;
      }
    }
    const fraction = (value - lower[0]) / (upper[0] - lower[0]);
    return lower[1].map((channel, index) =>
      Math.round(channel + (upper[1][index] - channel) * fraction),
    );
  }

  function drawHeatmapSamples(size) {
    for (const sample of samples) {
      const x = sample.x * size;
      const y = (1 - sample.y) * size;
      heatmapContext.beginPath();
      heatmapContext.arc(x, y, 5.5, 0, Math.PI * 2);
      heatmapContext.fillStyle = sample.target === 0 ? "#101426" : "#ffffff";
      heatmapContext.fill();
      heatmapContext.lineWidth = 1.3;
      heatmapContext.strokeStyle = "#f4f3fa";
      heatmapContext.stroke();
    }
  }

  function drawLossChart() {
    const width = lossWidth;
    const height = lossHeight;
    if (!width || !height) return;
    const left = 32;
    const right = width - 7;
    const top = 7;
    const bottom = height - 14;
    const chartWidth = right - left;
    const chartHeight = bottom - top;
    lossContext.clearRect(0, 0, width, height);

    lossContext.font = '8px "DM Mono", monospace';
    lossContext.textAlign = "right";
    for (let index = 0; index <= 2; index += 1) {
      const value = 0.5 - index * 0.25;
      const y = top + (chartHeight * index) / 2;
      lossContext.strokeStyle = "rgba(103, 120, 146, .15)";
      lossContext.lineWidth = 1;
      lossContext.beginPath();
      lossContext.moveTo(left, y);
      lossContext.lineTo(right, y);
      lossContext.stroke();
      lossContext.fillStyle = "#77849a";
      lossContext.fillText(value.toFixed(2), left - 6, y + 3);
    }

    if (!lossHistory.length) return;
    const points = lossHistory.map((loss, index) => ({
      x: left + (lossHistory.length === 1 ? 0 : (chartWidth * index) / (lossHistory.length - 1)),
      y: top + chartHeight * (1 - Math.min(loss, 0.5) / 0.5),
    }));

    lossContext.beginPath();
    lossContext.moveTo(points[0].x, bottom);
    for (const point of points) lossContext.lineTo(point.x, point.y);
    lossContext.lineTo(points[points.length - 1].x, bottom);
    lossContext.closePath();
    const fill = lossContext.createLinearGradient(0, top, 0, bottom);
    fill.addColorStop(0, "rgba(157, 125, 255, .22)");
    fill.addColorStop(1, "rgba(157, 125, 255, 0)");
    lossContext.fillStyle = fill;
    lossContext.fill();

    lossContext.beginPath();
    points.forEach((point, index) => {
      if (index === 0) lossContext.moveTo(point.x, point.y);
      else lossContext.lineTo(point.x, point.y);
    });
    lossContext.strokeStyle = "#765bd0";
    lossContext.lineWidth = 2;
    lossContext.shadowBlur = 8;
    lossContext.shadowColor = "rgba(118, 91, 208, .25)";
    lossContext.stroke();
    lossContext.shadowBlur = 0;
    const latest = points[points.length - 1];
    lossContext.beginPath();
    lossContext.arc(latest.x, latest.y, 3, 0, Math.PI * 2);
    lossContext.fillStyle = "#5940b3";
    lossContext.fill();
  }

  function updateExamples() {
    let correct = 0;
    for (let index = 0; index < samples.length; index += 1) {
      const prediction = predict(samples[index].x, samples[index].y).output;
      const isCorrect = (prediction >= 0.5 ? 1 : 0) === samples[index].target;
      if (isCorrect) correct += 1;
      const row = trainingRows[index];
      row.querySelector(".prediction-cell").textContent = prediction.toFixed(2);
      const result = row.querySelector(".result-cell");
      result.classList.toggle("correct", isCorrect);
      result.classList.toggle("incorrect", !isCorrect && step > 0);
      result.lastElementChild.textContent = step === 0 ? "—" : isCorrect ? "Correct" : "Learning";
    }
    accuracyValue.innerHTML = `${Math.round((correct / samples.length) * 100)}<span>%</span>`;
    accuracyCaption.textContent = `${correct} of 4 examples correct`;
  }

  function updateMetrics() {
    const loss = getLoss();
    stepValue.innerHTML = `${step.toLocaleString()} <span>/ ${TRAINING_STEPS.toLocaleString()}</span>`;
    stepCaption.textContent = step === 0 ? "Waiting to start" : `${Math.round((step / TRAINING_STEPS) * 100)}% complete`;
    lossValue.textContent = loss.toFixed(4);
    chartLoss.textContent = loss.toFixed(4);
    updateExamples();
  }

  function drawAll(timestamp = 0) {
    drawNetwork(timestamp);
    drawHeatmap();
    drawLossChart();
    updateMetrics();
  }

  function startTraining() {
    if (animationId !== null) {
      pauseTraining();
      return;
    }
    if (step === 0 || step >= TRAINING_STEPS) {
      model = createNetwork();
      step = 0;
      lossHistory = [getLoss()];
    }
    statusLabel.textContent = "Training";
    statusLabel.parentElement.classList.add("is-training");
    startButton.innerHTML = '<span class="button-icon pause-icon" aria-hidden="true">Ⅱ</span><span>Pause training</span>';
    startButton.setAttribute("aria-label", "Pause training");
    stepCaption.textContent = "Learning in progress";
    lastFrameTime = 0;
    animationId = requestAnimationFrame(trainFrame);
  }

  function pauseTraining() {
    if (animationId !== null) cancelAnimationFrame(animationId);
    animationId = null;
    statusLabel.textContent = "Paused";
    statusLabel.parentElement.classList.remove("is-training");
    startButton.innerHTML = '<span class="button-icon" aria-hidden="true">▶</span><span>Resume training</span>';
    startButton.setAttribute("aria-label", "Resume training");
    stepCaption.textContent = "Paused at current step";
    drawAll(performance.now());
  }

  function finishTraining() {
    animationId = null;
    statusLabel.textContent = "Training complete";
    statusLabel.parentElement.classList.remove("is-training");
    startButton.innerHTML = '<span class="button-icon" aria-hidden="true">↻</span><span>Train again</span>';
    startButton.setAttribute("aria-label", "Train again from the beginning");
    stepCaption.textContent = "All 6,000 steps complete";
    drawAll(performance.now());
  }

  function trainFrame(timestamp) {
    if (lastFrameTime && timestamp - lastFrameTime < 28) {
      animationId = requestAnimationFrame(trainFrame);
      return;
    }
    lastFrameTime = timestamp;

    const stepsPerFrame = prefersReducedMotion.matches ? 35 : 18;
    for (let index = 0; index < stepsPerFrame && step < TRAINING_STEPS; index += 1) {
      trainStep();
      step += 1;
      if (step % FRAME_INTERVAL === 0 || step === TRAINING_STEPS) lossHistory.push(getLoss());
    }

    drawAll(timestamp);
    if (step >= TRAINING_STEPS) finishTraining();
    else animationId = requestAnimationFrame(trainFrame);
  }

  function advanceOneStep() {
    if (animationId !== null) pauseTraining();
    if (step >= TRAINING_STEPS) resetTraining();
    trainStep();
    step += 1;
    if (step % FRAME_INTERVAL === 0 || lossHistory.length === 1) lossHistory.push(getLoss());
    statusLabel.textContent = "Stepping";
    stepCaption.textContent = "Advanced one training step";
    drawAll(performance.now());
  }

  function resetTraining() {
    if (animationId !== null) cancelAnimationFrame(animationId);
    animationId = null;
    model = createNetwork();
    step = 0;
    lossHistory = [getLoss()];
    statusLabel.textContent = "Ready to train";
    statusLabel.parentElement.classList.remove("is-training");
    startButton.innerHTML = '<span class="button-icon" aria-hidden="true">▶</span><span>Start training</span>';
    startButton.setAttribute("aria-label", "Start training the neural network");
    drawAll();
  }

  document.querySelector("#step-button").addEventListener("click", advanceOneStep);
  document.querySelector("#reset-button").addEventListener("click", resetTraining);
  heatmapCanvas.addEventListener("pointerdown", (event) => {
    surfaceDragX = event.clientX;
    heatmapCanvas.setPointerCapture(event.pointerId);
  });
  heatmapCanvas.addEventListener("pointermove", (event) => {
    if (surfaceDragX === null) return;
    surfaceYaw += (event.clientX - surfaceDragX) * 0.012;
    surfaceDragX = event.clientX;
    drawHeatmap();
  });
  heatmapCanvas.addEventListener("pointerup", () => { surfaceDragX = null; });
  heatmapCanvas.addEventListener("pointercancel", () => { surfaceDragX = null; });
  startButton.addEventListener("click", startTraining);
  document.querySelector("#help-button").addEventListener("click", () => {
    document.querySelector("#help-dialog").showModal();
  });
  window.addEventListener("resize", () => {
    measureCanvases();
    drawAll(performance.now());
  });
  window.addEventListener("beforeunload", () => {
    if (animationId !== null) cancelAnimationFrame(animationId);
  });

  model = createNetwork();
  lossHistory = [getLoss()];
  measureCanvases();
  drawAll();
})();
