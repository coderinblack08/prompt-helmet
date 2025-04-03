document.addEventListener("DOMContentLoaded", () => {
  const classifierToggle = document.getElementById("classifierToggle");
  const toggleStatus = document.getElementById("toggleStatus");
  const userPrompt = document.getElementById("userPrompt");
  const sendButton = document.getElementById("sendButton");
  const responseArea = document.getElementById("responseArea");
  const analysisContainer = document.getElementById("analysisContainer");
  const analysisContent = document.getElementById("analysisContent");
  const exampleButtons = document.querySelectorAll(".example-btn");

  // Toggle classifier status
  classifierToggle.addEventListener("change", () => {
    if (classifierToggle.checked) {
      toggleStatus.textContent = "ON";
      toggleStatus.style.color = "var(--accent-color)";
    } else {
      toggleStatus.textContent = "OFF";
      toggleStatus.style.color = "var(--danger-color)";
    }
  });

  // Send prompt to API
  async function sendPrompt(prompt) {
    // Show loading state
    responseArea.textContent = "Thinking...";
    responseArea.classList.remove("harmful", "safe");
    analysisContainer.classList.add("hidden");

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt: prompt,
          use_classifier: classifierToggle.checked,
        }),
      });

      const data = await response.json();

      // Display response
      responseArea.textContent = data.response;

      // Show analysis if harmful
      if (data.is_harmful) {
        responseArea.classList.add("harmful");
        analysisContainer.classList.remove("hidden");
        analysisContent.innerHTML = `
                    <p><strong>Detected potential prompt injection!</strong></p>
                    <p>The embedding model detected semantic similarity to a known harmful pattern.</p>
                    <p>Similarity score: ${(data.similarity * 100).toFixed(
                      2
                    )}%</p>
                    <p>Matched with harmful pattern: "${
                      data.matched_prompt
                    }"</p>
                    <p><small>Using embedding model: all-MiniLM-L6-v2 | LLM: qwen2.5:1.5b</small></p>
                `;
      } else {
        responseArea.classList.add("safe");

        // If classifier is off, show a note
        if (!classifierToggle.checked) {
          analysisContainer.classList.remove("hidden");
          analysisContent.innerHTML = `
                        <p><strong>Note:</strong> Embedding classifier is turned off. 
                        The model responded directly without checking for prompt injections.</p>
                        <p><small>Using LLM: qwen2.5:1.5b</small></p>
                    `;
        } else {
          // If classifier is on but found no issues
          analysisContainer.classList.remove("hidden");
          analysisContent.innerHTML = `
                        <p><strong>No injection detected.</strong> The embedding model did not find significant 
                        similarity to known harmful patterns.</p>
                        <p><small>Using embedding model: all-MiniLM-L6-v2 | LLM: qwen2.5:1.5b</small></p>
                    `;
        }
      }
    } catch (error) {
      responseArea.textContent = `Error: ${error.message}`;
      responseArea.classList.add("harmful");
    }
  }

  // Handle send button click
  sendButton.addEventListener("click", () => {
    const prompt = userPrompt.value.trim();
    if (prompt) {
      sendPrompt(prompt);
    }
  });

  // Handle Enter key in textarea
  userPrompt.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const prompt = userPrompt.value.trim();
      if (prompt) {
        sendPrompt(prompt);
      }
    }
  });

  // Handle example buttons
  exampleButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const examplePrompt = button.getAttribute("data-prompt");
      userPrompt.value = examplePrompt;
      sendPrompt(examplePrompt);
    });
  });
});
