import { useState } from "react";
import axios from "axios";

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);

  const API_URL = "https://6g05kvxgm9.execute-api.us-east-1.amazonaws.com/prod/submit";

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResponse("");

    try {
      const res = await axios.post(
        API_URL,
        { message: prompt },
        { headers: { "Content-Type": "application/json" } }
      );

      setResponse(res.data.response || "No response received");
    } catch (err) {
      console.error(err);
      setResponse("Error: " + (err.response?.data?.error || err.message));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 600, margin: "50px auto", fontFamily: "Arial" }}>
      <h1>Morgan Stanley AI Agent - Hackathon POC # 17 (Create/Dispatch Case - For creating a case - include "case"/"problem"/"ticket"/"issue" in the prompt)</h1>
      <form onSubmit={handleSubmit}>
        <textarea
          placeholder="Enter your message..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          style={{ width: "100%", height: 120, padding: 10, fontSize: 16 }}
        />
        <button
          type="submit"
          style={{ marginTop: 10, padding: "10px 20px", fontSize: 16 }}
          disabled={loading}
        >
          {loading ? "Sending..." : "Send"}
        </button>
      </form>

      {response && (
        <div
          style={{
            marginTop: 20,
            padding: 15,
            border: "1px solid #ddd",
            borderRadius: 5,
            backgroundColor: "#f9f9f9",
            whiteSpace: "pre-wrap",
          }}
        >
          {response}
        </div>
      )}
    </div>
  );
}
