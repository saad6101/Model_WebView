<script lang="ts">
  import { onMount } from "svelte";
  import { loadPyodide } from "pyodide";
  import Canvas from "./canvas.svelte";
  import { Button } from "$lib/components/ui/button/index.js";
  let modelReady = $state(false)
  let pyodide : any = $state();
  let result : any = $state("")
  let languages = 
  [
      { name : "Python"},
  ];
  let modelsPython = [
      { name: "Digit Model", file: "/models/model.npz" },
      { name: "Digit Model 2", file: "/models//mnist_model.npz" }
  ];
  let modelsRust = [
      { name: "coming_soon", file: "/models/model.npz" },
  ];
  let modelGo = [
      { name: "coming_soon", file: "/models/model.npz" },
  ];
  let model = 
  [
    { elements : modelsPython, loadFunction : loadModelPython, initalFunction : initializePython, loaded : false},
  ]
  let selectedModelIndex = $state(0);
  let selectedLanguageIndex = $state(0);
  async function loadModelPython(index: number) {
    selectedModelIndex = index;
    if (!pyodide) return;
    const modelFile = modelsPython[index].file;
    const response = await (await fetch(modelFile)).arrayBuffer();
    pyodide.FS.writeFile("model.npz", new Uint8Array(response));
    await pyodide.runPythonAsync(`
      import numpy as np
      params = np.load("model.npz")
      W1, b1 = params["W1"], params["b1"]
      W2, b2 = params["W2"], params["b2"]
      W3, b3 = params["W3"], params["b3"]

      def elu(Z, alpha=1.0):
          return np.where(Z > 0, Z, alpha * (np.exp(Z) - 1))

      def elu_deriv(Z, alpha=1):
          return np.where(Z > 0, 1, alpha * np.exp(Z))

      def forward_prop(W1, b1, W2, b2, W3, b3, X):
          Z1 = W1.dot(X) + b1
          A1 = elu(Z1)
          Z2 = W2.dot(A1) + b2
          A2 = elu(Z2)
          Z3 = W3.dot(A2) + b3
          A3 = np.exp(Z3) / np.sum(np.exp(Z3))
          return Z1, A1, Z2, A2, Z3, A3

      def get_predictions(A3):
          return np.argmax(A3, 0)

      def make_predictions(X):
          _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
          return get_predictions(A3)`);
    modelReady = true;
    }
  async function initializePython()
  {
      if (model[selectedLanguageIndex].loaded == true) {return}
      pyodide = await loadPyodide({ indexURL: "/pyodide/" })
      await pyodide.loadPackage("numpy");
      model[selectedLanguageIndex].loaded == true;
  }

  onMount(
    async () => {
      await model[selectedLanguageIndex].initalFunction()
      await model[selectedLanguageIndex].loadFunction(selectedModelIndex);
      modelReady = true;
    }
  );
    let pixels: number[][] = $state(Array.from({ length: 28 }, () =>
      Array(28).fill(0)
  ));
async function predict() {
  const flatPixels = pixels.flat().map(p => p / 255); 
  if (!pyodide) return;
  await pyodide.globals.set("pixels", flatPixels);
  const pred = await pyodide.runPythonAsync(`
      import numpy as np
      x = np.array(pixels, dtype=np.float32).reshape(784,1)
      pred = make_predictions(x)
      int(pred)  # convert numpy int to Python int
  `);
  alert("Predicted digit: " + pred);
  result = pred
}
</script>
<h1 class="flex items-center justify-center pt-2 font-bold  text-4xl">   
  <select bind:value={selectedLanguageIndex} onchange={() => {selectedModelIndex = 0 }}>
    {#each languages as language, i}
      <option class=" text-black text-lg" value={i}>{language.name}</option>
    {/each}
  </select> 
</h1>
<div class="flex justify-center mb-5">
  <select bind:value={selectedModelIndex} onchange={() => model[selectedLanguageIndex].loadFunction(selectedModelIndex)}>
    {#each model[selectedLanguageIndex].elements as models, i}
      <option class=" text-black outline-none" value={i}>{models.name}</option>
    {/each}
  </select>
</div>
<div class="flex items-center justify-center pr-108">Result = {result}</div>
<div class="flex items-center justify-center pt-5"><Canvas bind:modelReady = {modelReady} bind:pixels =  {pixels} bind:result = {result}></Canvas></div>
<div class="flex items-center justify-center pt-3 ">
  <Button disabled={!modelReady} onclick= {predict}>Predict</Button>
</div>