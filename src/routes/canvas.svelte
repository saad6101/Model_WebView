<script lang="ts">
  let isDrawing = false;
  let animation = $state(false)
  let { pixels = $bindable(), result = $bindable(), modelReady = $bindable(), radius = $bindable() }: { pixels: number[][], result : any, modelReady : boolean, radius : number  } = $props();
  
  function togglePixel(row: number, col: number) {
    if (!modelReady) { return; }
    pixels = pixels.map((r, ri) =>
      r.map((c, ci) => {
        const distance = Math.sqrt((ri - row) ** 2 + (ci - col) ** 2);
        if (distance <= radius) {
          const intensity = Math.max(0, 255 - (distance / radius) * 255);
          return Math.max(c, intensity); 
        }
        return c;
      })
    );
  }

  function clearGrid() {
    if (!modelReady) {return}
    pixels = Array.from({ length: 28 }, () =>
      Array(28).fill(0)
    );
    result = ""
    animation = true
    setTimeout(() => animation = false, 500);
  }
  function handleDown(row: number, col: number) {
    if (!modelReady) {return}
    isDrawing = true;
    togglePixel(row, col);
  }

  function handleEnter(row: number, col: number) {
    if (!modelReady) {return}
    if (isDrawing) {
      togglePixel(row, col);
    }
  }

  function handleUp() {
    if (!modelReady) {return}
    isDrawing = false;
  }

</script>

<style>
  .grid {
    display: grid;
    grid-template-columns: repeat(28, 15px);
    grid-template-rows: repeat(28, 15px);
    user-select: none;
  }
  .pixel {
    width: 15px;
    height: 15px;
    padding: 0;
    border: 1px solid #444;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 10px;
    color: white;
  }
</style>

<div
  class="grid"
  onmouseup={handleUp}
  onmouseleave={handleUp}
>
  {#each pixels as row, rowIndex}
    {#each row as value, colIndex}
        <div
        class="pixel {modelReady ? "" : "cursor-not-allowed"}"
        style="background-color: rgb({255 - value}, {255 - value}, {255 - value}) "
        onmousedown={() => handleDown(rowIndex, colIndex)}
        onmouseenter={() => handleEnter(rowIndex, colIndex)}
      >
        {value > 0 ? Math.round(value) : ''}
      </div>
    {/each}
  {/each}
</div>

<div style="">
  <button onclick={clearGrid} disabled={animation}
  class="pl-5 transition-opacity duration-500 
  {animation   ? 
  'opacity-50 cursor-not-allowed'   :   ''
  }">
    <img src="/material.png" 
    alt="refresh" 
    width= "30" height="30" 
    class="transition-transform duration-500 {animation ? 'animate-spin' : ''}"/>
    <div class="flex items-center justify-center"><a href="/" download="model.py"><img class="flex" width=30 height="30" src="download.png" alt="download"></a></div>
  </button>
</div>
