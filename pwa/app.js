const form = document.getElementById("configForm");
const output = document.getElementById("configOutput");
const previewCard = document.getElementById("previewCard");
const previewWord = document.getElementById("previewWord");
const previewPhonetic = document.getElementById("previewPhonetic");
const previewSynonyms = document.getElementById("previewSynonyms");
const previewLang = document.getElementById("previewLang");
const previewMode = document.getElementById("previewMode");
const previewModel = document.getElementById("previewModel");
const previewLayout = document.getElementById("previewLayout");

const exportButton = document.getElementById("exportButton");
const copyButton = document.getElementById("copyButton");
const resetButton = document.getElementById("resetButton");
const installButton = document.getElementById("installButton");

const STORAGE_KEY = "wordscard-config";

let deferredPrompt = null;

function getFormData() {
  const data = new FormData(form);
  const config = Object.fromEntries(data.entries());
  config.padding = Number(config.padding);
  config.wordSize = Number(config.wordSize);
  config.phoneticSize = Number(config.phoneticSize);

  config.language = {
    primary: config.primaryLanguage,
    secondary: config.secondaryLanguage,
    contentMode: config.contentMode,
    chineseStrategy: config.chineseStrategy,
  };

  config.ai = {
    mode: config.aiMode,
    primaryProvider: config.primaryProvider,
    secondaryProvider: config.secondaryProvider,
    openaiModel: config.openaiModel,
    deepseekModel: config.deepseekModel,
    fallbackPolicy: config.fallbackPolicy,
  };

  config.layout = {
    preset: config.layoutPreset,
    ratio: config.cardRatio,
    padding: config.padding,
    accentColor: config.accentColor,
    headingFont: config.headingFont,
    bodyFont: config.bodyFont,
    wordSize: config.wordSize,
    phoneticSize: config.phoneticSize,
  };

  config.sample = {
    word: config.sampleWord,
    phonetic: config.samplePhonetic,
    synonyms: [config.synonymOne, config.synonymTwo].filter(Boolean),
  };

  return config;
}

function updatePreview(config) {
  previewWord.textContent = config.sample.word || "";
  previewPhonetic.textContent = config.sample.phonetic || "";

  previewSynonyms.innerHTML = "";
  config.sample.synonyms.forEach((syn) => {
    const span = document.createElement("span");
    span.textContent = syn;
    previewSynonyms.appendChild(span);
  });

  const langLabel = `${config.language.primary.toUpperCase()}${
    config.language.secondary !== "none"
      ? ` + ${config.language.secondary.toUpperCase()}`
      : ""
  }`;
  previewLang.textContent = langLabel;
  previewMode.textContent = config.ai.mode;

  const modelLabel = `${
    config.ai.mode === "openai"
      ? config.ai.openaiModel
      : config.ai.mode === "deepseek"
      ? config.ai.deepseekModel
      : `${config.ai.openaiModel} → ${config.ai.deepseekModel}`
  }`;
  previewModel.textContent = modelLabel;
  previewLayout.textContent = config.layout.preset
    .split("-")
    .map((s) => s[0].toUpperCase() + s.slice(1))
    .join(" ");

  previewCard.style.setProperty("--accent", config.layout.accentColor);
  previewCard.style.padding = `${config.layout.padding}px`;
  previewCard.style.fontFamily = config.layout.bodyFont;
  previewWord.style.fontFamily = config.layout.headingFont;
  previewWord.style.fontSize = `${config.layout.wordSize}px`;
  previewPhonetic.style.fontSize = `${config.layout.phoneticSize}px`;
}

function updateOutput(config) {
  output.value = JSON.stringify(config, null, 2);
}

function saveConfig(config) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
}

function loadConfig() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function applyConfig(config) {
  if (!config) return;
  Object.entries(config).forEach(([key, value]) => {
    const field = form.elements[key];
    if (field) field.value = value;
  });

  if (config.language) {
    form.elements.primaryLanguage.value = config.language.primary;
    form.elements.secondaryLanguage.value = config.language.secondary;
    form.elements.contentMode.value = config.language.contentMode;
    form.elements.chineseStrategy.value = config.language.chineseStrategy;
  }

  if (config.ai) {
    form.elements.aiMode.value = config.ai.mode;
    form.elements.primaryProvider.value = config.ai.primaryProvider;
    form.elements.secondaryProvider.value = config.ai.secondaryProvider;
    form.elements.openaiModel.value = config.ai.openaiModel;
    form.elements.deepseekModel.value = config.ai.deepseekModel;
    form.elements.fallbackPolicy.value = config.ai.fallbackPolicy;
  }

  if (config.layout) {
    form.elements.layoutPreset.value = config.layout.preset;
    form.elements.cardRatio.value = config.layout.ratio;
    form.elements.padding.value = config.layout.padding;
    form.elements.accentColor.value = config.layout.accentColor;
    form.elements.headingFont.value = config.layout.headingFont;
    form.elements.bodyFont.value = config.layout.bodyFont;
    form.elements.wordSize.value = config.layout.wordSize;
    form.elements.phoneticSize.value = config.layout.phoneticSize;
  }

  if (config.sample) {
    form.elements.sampleWord.value = config.sample.word || "";
    form.elements.samplePhonetic.value = config.sample.phonetic || "";
    form.elements.synonymOne.value = config.sample.synonyms?.[0] || "";
    form.elements.synonymTwo.value = config.sample.synonyms?.[1] || "";
  }
}

function handleChange() {
  const config = getFormData();
  updatePreview(config);
  updateOutput(config);
  saveConfig(config);
}

form.addEventListener("input", handleChange);

exportButton.addEventListener("click", () => {
  const config = getFormData();
  const blob = new Blob([JSON.stringify(config, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "wordscard-config.json";
  link.click();
  URL.revokeObjectURL(url);
});

copyButton.addEventListener("click", async () => {
  const config = getFormData();
  try {
    await navigator.clipboard.writeText(JSON.stringify(config, null, 2));
    copyButton.textContent = "Copied!";
    setTimeout(() => (copyButton.textContent = "Copy JSON"), 1200);
  } catch {
    copyButton.textContent = "Copy failed";
    setTimeout(() => (copyButton.textContent = "Copy JSON"), 1200);
  }
});

resetButton.addEventListener("click", () => {
  localStorage.removeItem(STORAGE_KEY);
  form.reset();
  handleChange();
});

window.addEventListener("beforeinstallprompt", (event) => {
  event.preventDefault();
  deferredPrompt = event;
  installButton.hidden = false;
});

installButton.addEventListener("click", async () => {
  if (!deferredPrompt) return;
  deferredPrompt.prompt();
  await deferredPrompt.userChoice;
  deferredPrompt = null;
  installButton.hidden = true;
});

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("service-worker.js");
  });
}

const saved = loadConfig();
if (saved) {
  applyConfig(saved);
}
handleChange();
