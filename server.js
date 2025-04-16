const express = require("express");
const cors = require("cors");
const path = require("path");
const app = express();

// Enable CORS for all routes
app.use(cors());

// Serve static files (including the WASM files)
app.use(express.static(path.join(__dirname, "public")));

// Optionally, add a route for your WASM files if they are stored in a subfolder
app.use("/mediapipe", express.static(path.join(__dirname, "public", "mediapipe")));

app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

const port = 3000;
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
