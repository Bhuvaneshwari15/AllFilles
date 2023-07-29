import React, { useState } from "react";
import "./App.css";
const App = () => {
const [counter, setCounter] = useState(0)
const handleClick1 = () => {
setCounter(counter + 1)
}
const handleClick2 = () => {
setCounter(counter - 1)
}
return (
<div>
Counter App
<div>
{counter}
</div>
<div>
<button onClick={handleClick1}>Increment</button>
<button onClick={handleClick2}>Decrement</button>
</div>
</div>
)
}
export default App