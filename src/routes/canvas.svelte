<script lang="ts">
  let isDrawing = false;
  let animation = $state(false)
  let { pixels = $bindable(), result = $bindable(), modelReady = $bindable() }: { pixels: number[][], result : any, modelReady : boolean  } = $props();
  

  function togglePixel(row: number, col: number) {
    if (!modelReady) {return}
    pixels = pixels.map((r, ri) =>
      ri === row
        ? r.map((c, ci) => (ci === col ? 255 : c))
        : r
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
        class="pixel"
        style="background-color: {value === 255 ? 'white' : 'black'}"
        onmousedown={() => handleDown(rowIndex, colIndex)}
        onmouseenter={() => handleEnter(rowIndex, colIndex)}
      >
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
  </button>
</div>
