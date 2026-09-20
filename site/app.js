/* Exu explainer interactions.
   Every number on the page comes from the same math the Python package uses:
   log score with a 1e-4 floor, spherical score, the composite reward, and the
   score-function policy update. No backend, no build step, no dependencies. */

(() => {
  "use strict";

  const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
  const LOG_FLOOR = 1e-4;

  /* ------------------------------------------------------------- score math */

  const logScore = (q, y) => {
    let total = 0;
    for (let i = 0; i < q.length; i += 1) {
      total += y[i] * Math.log(Math.max(q[i], LOG_FLOOR));
    }
    return total;
  };

  const spherical = (q, y) => {
    let dot = 0;
    let norm = 0;
    for (let i = 0; i < q.length; i += 1) {
      dot += q[i] * y[i];
      norm += q[i] * q[i];
    }
    return dot / Math.max(Math.sqrt(norm), 1e-12);
  };

  const composite = (q, y, weights = { spherical: 0.75, rps: 1.0 }, ordinal = false) => {
    let value = logScore(q, y) + weights.spherical * spherical(q, y);
    if (ordinal) {
      let cumulativeQ = 0;
      let cumulativeY = 0;
      let error = 0;
      for (let i = 0; i < q.length; i += 1) {
        cumulativeQ += q[i];
        cumulativeY += y[i];
        error += (cumulativeQ - cumulativeY) ** 2;
      }
      value -= weights.rps * (error / Math.max(q.length - 1, 1));
    }
    return value;
  };

  const softmax = (logits, temperature = 1) => {
    const scaled = logits.map((value) => value / temperature);
    const peak = Math.max(...scaled);
    const exponentials = scaled.map((value) => Math.exp(value - peak));
    const total = exponentials.reduce((sum, value) => sum + value, 0);
    return exponentials.map((value) => value / total);
  };

  const normal = (() => {
    let spare = null;
    return () => {
      if (spare !== null) {
        const value = spare;
        spare = null;
        return value;
      }
      let u = 0;
      let v = 0;
      while (u === 0) u = Math.random();
      while (v === 0) v = Math.random();
      const magnitude = Math.sqrt(-2 * Math.log(u));
      spare = magnitude * Math.sin(2 * Math.PI * v);
      return magnitude * Math.cos(2 * Math.PI * v);
    };
  })();

  /* -------------------------------------------------------------- hero bars */

  function initHero() {
    const container = document.getElementById("hero-bars");
    if (!container) return;
    const weights = [1.1, 0.4, 2.0, 3.2, 1.6, 0.6];
    const bars = weights.map(() => {
      const bar = document.createElement("span");
      container.appendChild(bar);
      return bar;
    });
    const draw = () => {
      const distribution = softmax(weights.map((weight) => weight + normal() * 0.9));
      const peak = Math.max(...distribution);
      bars.forEach((bar, index) => {
        bar.style.height = `${18 + (distribution[index] / peak) * 82}%`;
      });
    };
    draw();
    if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      window.setInterval(draw, 2200);
    }
  }

  /* ---------------------------------------------------------- reward demo */

  function initReward() {
    const slider = document.getElementById("reward-slider");
    const line = document.getElementById("reward-line");
    const step = document.getElementById("reward-step");
    const area = document.getElementById("reward-area");
    const cursor = document.getElementById("reward-cursor");
    const node = document.getElementById("reward-node");
    const grid = document.querySelector("#failure .chart-grid");
    const optimum = document.getElementById("reward-optimum");
    if (!slider || !line || !grid) return;

    const padLeft = 46;
    const padTop = 18;
    const plotWidth = 640 - padLeft - 18;
    const plotHeight = 210;
    const target = [0.7, 0.3];
    const honest = 0.7;

    const samples = 220;
    const curve = [];
    for (let index = 0; index <= samples; index += 1) {
      const confidence = 0.05 + (index / samples) * 0.945;
      const q = [confidence, 1 - confidence];
      curve.push({ confidence, log: logScore(q, target), spherical: spherical(q, target) });
    }
    const logMin = Math.min(...curve.map((point) => point.log));
    const logMax = Math.max(...curve.map((point) => point.log));
    const sphMin = Math.min(...curve.map((point) => point.spherical));
    const sphMax = Math.max(...curve.map((point) => point.spherical));

    const xAt = (confidence) => padLeft + ((confidence - 0.05) / 0.945) * plotWidth;
    const yAt = (value) => padTop + (1 - value) * plotHeight;
    const normalise = (value, low, high) => (value - low) / (high - low);

    // grid
    for (let index = 0; index <= 4; index += 1) {
      const value = index / 4;
      const y = yAt(value);
      const gridLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
      gridLine.setAttribute("x1", padLeft);
      gridLine.setAttribute("x2", padLeft + plotWidth);
      gridLine.setAttribute("y1", y);
      gridLine.setAttribute("y2", y);
      grid.appendChild(gridLine);
      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("x", 6);
      label.setAttribute("y", y + 3);
      label.textContent = value === 1 ? "best" : value.toFixed(2);
      grid.appendChild(label);
    }
    [0.05, 0.25, 0.5, 0.75, 1].forEach((confidence) => {
      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("x", xAt(confidence));
      label.setAttribute("y", padTop + plotHeight + 20);
      label.setAttribute("text-anchor", "middle");
      label.textContent = confidence.toFixed(2);
      grid.appendChild(label);
    });

    const logPath = curve
      .map((point, index) => {
        const x = xAt(point.confidence);
        const y = yAt(normalise(point.log, logMin, logMax));
        return `${index === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
      })
      .join(" ");
    line.setAttribute("d", logPath);

    const stepPath = [
      `M${xAt(0.05)} ${yAt(0)}`,
      `L${xAt(0.5)} ${yAt(0)}`,
      `L${xAt(0.5)} ${yAt(1)}`,
      `L${xAt(0.995)} ${yAt(1)}`,
    ].join(" ");
    step.setAttribute("d", stepPath);

    const honestX = xAt(honest);
    area.setAttribute(
      "d",
      `${logPath} L${xAt(0.995)} ${yAt(-0.1)} L${xAt(0.05)} ${yAt(-0.1)} Z`
    );
    optimum.setAttribute("x", honestX + 6);
    optimum.setAttribute("y", yAt(1) - 6);
    optimum.textContent = "honest optimum 0.70";

    const logMaxValue = logMax;

    const render = () => {
      const confidence = Number(slider.value) / 100;
      const q = [confidence, 1 - confidence];
      const x = xAt(confidence);
      const y = yAt(normalise(logScore(q, target), logMin, logMax));
      cursor.setAttribute("x1", x);
      cursor.setAttribute("x2", x);
      cursor.setAttribute("y1", yAt(0));
      cursor.setAttribute("y2", yAt(1));
      node.setAttribute("cx", x);
      node.setAttribute("cy", y);
      node.classList.toggle("bad", confidence > 0.9);
      document.getElementById("reward-value").textContent = confidence.toFixed(2);
      document.getElementById("stat-log").textContent = logScore(q, target).toFixed(3);
      document.getElementById("stat-sph").textContent = spherical(q, target).toFixed(3);
      document.getElementById("stat-bin").textContent = confidence > 0.5 ? "1.000" : "0.000";
      const gap = Math.max(0, logMaxValue - logScore(q, target));
      document.getElementById("reward-caption").textContent =
        confidence > 0.72
          ? `At ${confidence.toFixed(2)} you are more certain than the 70% you actually know. ` +
            `The proper score is ${logScore(q, target).toFixed(3)} instead of the ${logMaxValue.toFixed(3)} available at 0.70, ` +
            `a cost of ${gap.toFixed(3)} nats. The 1-or-0 reward cannot see the difference.`
          : `At ${confidence.toFixed(2)} the proper score is ${logScore(q, target).toFixed(3)}. ` +
            `The best honest report is 0.70, worth ${logMaxValue.toFixed(3)}. ` +
            `The dashed 1-or-0 reward is flat everywhere above 0.50, so it rewards being sure.`;
    };

    slider.addEventListener("input", render);
    render();
  }

  /* ------------------------------------------------------- primitives tabs */

  const PRIMITIVES = {
    choice: {
      state: '"I was charged twice for the same invoice."',
      question: "kind: choice · where should this ticket go?",
      options: [
        { name: "billing", value: 0.86 },
        { name: "support", value: 0.11 },
        { name: "security", value: 0.03 },
      ],
    },
    score: {
      state: '"Production is down for every customer."',
      question: "kind: score · how urgent is this?",
      options: [
        { name: "Level 0: low", value: 0.02 },
        { name: "Level 1: normal", value: 0.08 },
        { name: "Level 2: high", value: 0.9 },
      ],
    },
    noul: {
      state: '"Please update the invoice address."',
      question: "kind: noul · does this describe a failed payment?",
      options: [
        { name: "No", value: 0.97 },
        { name: "Yes", value: 0.03 },
      ],
    },
  };

  function renderDistribution(container, options) {
    container.textContent = "";
    options.forEach((option) => {
      const row = document.createElement("div");
      row.className = "dist-row";
      const name = document.createElement("span");
      name.className = "dist-name";
      name.textContent = option.name;
      const track = document.createElement("div");
      track.className = "dist-track";
      const fill = document.createElement("div");
      fill.className = "dist-fill";
      fill.style.width = `${option.value * 100}%`;
      track.appendChild(fill);
      const value = document.createElement("span");
      value.className = "dist-value";
      value.textContent = option.value.toFixed(2);
      row.append(name, track, value);
      container.appendChild(row);
    });
  }

  function initPrimitives() {
    const tabs = [...document.querySelectorAll(".tab")];
    const distance = document.getElementById("prim-dist");
    if (!tabs.length || !distance) return;
    const select = (kind) => {
      tabs.forEach((tab) => tab.setAttribute("aria-selected", String(tab.dataset.kind === kind)));
      const data = PRIMITIVES[kind];
      document.getElementById("prim-state").textContent = data.state;
      document.getElementById("prim-question").textContent = data.question;
      renderDistribution(distance, data.options);
    };
    tabs.forEach((tab) => tab.addEventListener("click", () => select(tab.dataset.kind)));
    select("choice");
  }

  /* ------------------------------------------------------------ tape demo */

  const SEQ = {
    maxLength: 512,
    headerBudget: 192,
    maxOptionTokens: 48,
    minInstructionTokens: 8,
    minOptionTextTokens: 3,
  };

  const TAPE = {
    choice: {
      prefix: "type: choice",
      instruction: "where should this ticket go",
      options: ["billing: payment invoice or refund", "support: access or outage", "security: possible fraud"],
      state: "card was charged twice for the same invoice",
    },
    score: {
      prefix: "type: ordinal scale",
      instruction: "how urgent is this ticket",
      options: ["level 0: low", "level 1: normal", "level 2: high"],
      state: "production is down for every customer",
    },
    noul: {
      prefix: "type: yes or no",
      instruction: "does this message describe a failed payment",
      options: ["no", "yes"],
      state: "please update the invoice address",
    },
  };

  // Approximate token count. The real builder uses the encoder tokenizer; the
  // budget arithmetic below is the same.
  const tokens = (text) => text.split(/\s+/).filter(Boolean);

  function allocate(optionCount, prefixLength, optionLengths) {
    const separatorCount = optionCount + 2;
    const fixed = 1 + prefixLength + separatorCount + optionCount;
    const available = SEQ.headerBudget - fixed;
    const minimum = SEQ.minInstructionTokens + optionCount * SEQ.minOptionTextTokens;
    if (available < minimum) {
      return { error: `too many options for a ${SEQ.headerBudget}-token header` };
    }
    const optionCap = SEQ.maxOptionTokens - 1;
    const optionTotal = Math.min(
      optionCount * optionCap,
      available - SEQ.minInstructionTokens
    );
    const base = Math.floor(optionTotal / optionCount);
    const remainder = optionTotal % optionCount;
    const optionLimits = Array.from({ length: optionCount }, (_value, index) =>
      base + (index < remainder ? 1 : 0)
    );
    const instructionLimit = available - optionLimits.reduce((sum, value) => sum + value, 0);
    return {
      optionLimits,
      instructionLimit,
      optionTextTotal: Math.min(
        optionLimits.reduce((sum, limit, index) => sum + Math.min(limit, optionLengths[index]), 0),
        optionTotal
      ),
    };
  }

  function tokenChip(text, kind) {
    const chip = document.createElement("span");
    chip.className = `token ${kind}`;
    chip.textContent = text;
    return chip;
  }

  function initSequence() {
    const tape = document.getElementById("seq-tape");
    const bar = document.getElementById("budget-bar");
    const caption = document.getElementById("seq-caption");
    const optionsSlider = document.getElementById("seq-options");
    const optionsValue = document.getElementById("seq-options-value");
    const segments = [...document.querySelectorAll(".seg")];
    if (!tape || !bar) return;

    let kind = "choice";

    const draw = () => {
      const data = TAPE[kind];
      const prefixes = tokens(data.prefix);
      const instruction = tokens(data.instruction);
      const count = kind === "noul" ? 2 : Number(optionsSlider.value);
      const optionTexts = Array.from({ length: count }, (_value, index) =>
        tokens(data.options[index % data.options.length])
      );
      const optionLengths = optionTexts.map((words) => words.length);
      const plan = allocate(count, prefixes.length, optionLengths);

      tape.textContent = "";
      const add = (text, className) => {
        tape.appendChild(tokenChip(text, className || ""));
      };
      add("[CLS]", "special");
      prefixes.forEach((word) => add(word, "label"));
      if (plan.error) {
        add(plan.error, "marker");
        bar.textContent = "";
        caption.textContent = `${plan.error}. Raise the header budget or reduce options. This is the failure the builder raises instead of answering wrongly in silence.`;
        return;
      }
      instruction.slice(0, plan.instructionLimit).forEach((word) => add(word, "instruction"));
      add("[SEP]", "special");

      let markerCount = 0;
      optionTexts.forEach((words, index) => {
        add("[MASK]", "marker");
        markerCount += 1;
        const limit = plan.optionLimits[index];
        words.slice(0, limit).forEach((word) => add(word, "option"));
        if (words.length > limit) add("…", "special");
        add("[SEP]", "special");
      });

      const headerLength =
        1 +
        prefixes.length +
        plan.instructionLimit +
        1 +
        optionTexts.reduce((sum, words, index) => sum + 1 + Math.min(words.length, plan.optionLimits[index]) + 1, 0);
      const stateLimit = Math.max(0, SEQ.maxLength - headerLength - 1);
      const stateWords = tokens(data.state);
      stateWords.slice(0, stateLimit).forEach((word) => add(word, "state"));
      add("[SEP]", "special");

      const instructionTokens = plan.instructionLimit;
      const optionTokens = optionTexts.reduce(
        (sum, words, index) => sum + Math.min(words.length, plan.optionLimits[index]),
        0
      );
      const scaffoldingTokens = prefixes.length + 1 + count + (count + 1);
      const shownState = Math.min(stateWords.length, stateLimit);
      const segmentsData = [
        { value: Math.max(instructionTokens, 1), color: "var(--signal)" },
        { value: Math.max(optionTokens, 1), color: "var(--warn)" },
        { value: Math.max(scaffoldingTokens, 1), color: "var(--dim)" },
        { value: Math.max(shownState, 1), color: "var(--panel-2)" },
      ];
      const total = segmentsData.reduce((sum, item) => sum + item.value, 0);
      bar.textContent = "";
      segmentsData.forEach((item) => {
        const segment = document.createElement("span");
        segment.style.width = `${(item.value / total) * 100}%`;
        segment.style.background = item.color;
        if (item.color === "var(--panel-2)") segment.style.border = "1px solid var(--line)";
        bar.appendChild(segment);
      });

      caption.textContent =
        `${count} options · ${plan.instructionLimit} instruction tokens · ` +
        `${plan.optionLimits[0]} to ${plan.optionLimits[count - 1]} tokens per option · ` +
        `state keeps ${stateLimit}. Options yield their share before the instruction drops below ` +
        `${SEQ.minInstructionTokens}. Token counts are word approximations here; the builder uses the encoder tokenizer.`;
    };

    segments.forEach((segment) =>
      segment.addEventListener("click", () => {
        kind = segment.dataset.kind;
        segments.forEach((other) =>
          other.setAttribute("aria-pressed", String(other === segment))
        );
        optionsSlider.disabled = kind === "noul";
        draw();
      })
    );
    optionsSlider.addEventListener("input", () => {
      optionsValue.textContent = optionsSlider.value;
      draw();
    });
    draw();
  }

  /* ---------------------------------------------------------- policy demo */

  function initPolicy() {
    const logitsContainer = document.getElementById("policy-logits");
    const candidates = document.getElementById("policy-candidates");
    const sampleButton = document.getElementById("policy-sample");
    const stepButton = document.getElementById("policy-step");
    const resetButton = document.getElementById("policy-reset");
    const tally = document.getElementById("policy-tally");
    if (!logitsContainer || !sampleButton) return;

    const target = [0.1, 0.8, 0.1];
    const names = ["A", "B", "C"];
    const initial = [0, 0, 0];
    const samplesPerStep = 8;
    let logits = [...initial];
    let step = 0;
    let drawn = null;

    const sigma = () => Math.max(0.15, 0.6 * Math.pow(0.85, step));

    const renderLogits = () => {
      const probabilities = softmax(logits);
      renderDistribution(
        logitsContainer,
        names.map((name, index) => ({ name, value: probabilities[index] }))
      );
      document.getElementById("stat-b").textContent = probabilities[1].toFixed(3);
    };

    const renderCandidates = () => {
      candidates.textContent = "";
      if (!drawn) {
        for (let index = 0; index < samplesPerStep; index += 1) {
          const placeholder = document.createElement("div");
          placeholder.className = "candidate";
          const mini = document.createElement("div");
          mini.className = "mini";
          for (let bar = 0; bar < 3; bar += 1) {
            const span = document.createElement("span");
            span.style.height = "18%";
            mini.appendChild(span);
          }
          placeholder.appendChild(mini);
          const reward = document.createElement("span");
          reward.className = "candidate-reward";
          reward.textContent = "n/a";
          placeholder.appendChild(reward);
          candidates.appendChild(placeholder);
        }
        return;
      }
      drawn.forEach((draw) => {
        const card = document.createElement("div");
        card.className = `candidate ${draw.advantage > 0 ? "win" : "lose"}`;
        const mini = document.createElement("div");
        mini.className = "mini";
        draw.probabilities.forEach((value) => {
          const span = document.createElement("span");
          span.style.height = `${Math.max(4, value * 100)}%`;
          mini.appendChild(span);
        });
        card.appendChild(mini);
        const reward = document.createElement("span");
        reward.className = "candidate-reward";
        reward.textContent = draw.reward.toFixed(2);
        card.appendChild(reward);
        candidates.appendChild(card);
      });
    };

    const sample = () => {
      const spread = sigma();
      const current = [...logits];
      drawn = Array.from({ length: samplesPerStep }, () => {
        const noise = [0, 1, 2].map(() => normal());
        const mean = noise.reduce((sum, value) => sum + value, 0) / noise.length;
        const projected = noise.map((value) => value - mean);
        const sampled = current.map((value, index) => value + spread * projected[index]);
        const probabilities = softmax(sampled);
        const reward = composite(probabilities, target);
        return { projected, probabilities, reward };
      });
      const rewards = drawn.map((draw) => draw.reward);
      const meanReward = rewards.reduce((sum, value) => sum + value, 0) / rewards.length;
      const deviation = Math.sqrt(
        rewards.reduce((sum, value) => sum + (value - meanReward) ** 2, 0) / rewards.length
      );
      drawn.forEach((draw) => {
        draw.advantage = (draw.reward - meanReward) / Math.max(deviation, 1e-8);
      });
      renderCandidates();
      document.getElementById("stat-reward").textContent = meanReward.toFixed(3);
      document.getElementById("stat-adv").textContent = deviation.toFixed(3);
      stepButton.disabled = false;
      sampleButton.textContent = "Resample";
    };

    const applyStep = () => {
      if (!drawn) return;
      const spread = sigma();
      const learningRate = 0.35;
      const update = [0, 0, 0];
      drawn.forEach((draw) => {
        draw.projected.forEach((value, index) => {
          update[index] += (draw.advantage * value) / spread;
        });
      });
      logits = logits.map((value, index) => value + (learningRate * update[index]) / drawn.length);
      step += 1;
      drawn = null;
      tally.textContent = `step ${step} · sigma ${sigma().toFixed(2)}`;
      stepButton.disabled = true;
      sampleButton.textContent = "Sample candidates";
      renderLogits();
      renderCandidates();
      document.getElementById("stat-reward").textContent = "n/a";
      document.getElementById("stat-adv").textContent = "n/a";
    };

    sampleButton.addEventListener("click", sample);
    stepButton.addEventListener("click", applyStep);
    resetButton.addEventListener("click", () => {
      logits = [...initial];
      step = 0;
      drawn = null;
      tally.textContent = "step 0";
      stepButton.disabled = true;
      sampleButton.textContent = "Sample candidates";
      renderLogits();
      renderCandidates();
      document.getElementById("stat-reward").textContent = "n/a";
      document.getElementById("stat-adv").textContent = "n/a";
    });

    renderLogits();
    renderCandidates();
  }

  /* ------------------------------------------------------------------ boot */

  // Exposed so the page's math can be exercised headlessly (tests, console).
  if (typeof window !== "undefined") {
    window.Exu = { logScore, spherical, composite, softmax };
  }

  if (typeof document !== "undefined") {
    document.addEventListener("DOMContentLoaded", () => {
      initHero();
      initReward();
      initPrimitives();
      initSequence();
      initPolicy();
    });
  }
})();
