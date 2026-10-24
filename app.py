"""Wakil AI application: draw a digit, generate text, see the models.

    pip install -r requirements-app.txt
    python app.py              # opens http://127.0.0.1:7860
    python app.py --share      # also creates a temporary public link (e.g. from Colab)

This file only contains the interface. All the AI lives in wakil_ai/ (see service.py).
"""
import argparse

import gradio as gr

from wakil_ai.service import WakilService

service = WakilService()


def read_drawing(editor_value):
    """Gradio's Sketchpad gives {'background', 'layers', 'composite'}; the model needs the composite image."""
    image = editor_value.get("composite") if isinstance(editor_value, dict) else editor_value
    probabilities, seen, message = service.read_digit(image)
    return probabilities, seen, message, seen          # the last output is remembered for "Teach"


def build_app():
    with gr.Blocks(title="Wakil AI") as app:
        gr.Markdown("# 🤖 Wakil AI\nNeural networks built and trained from scratch with NumPy.")

        with gr.Tab("✍️ Read a digit"):
            gr.Markdown("1. Click the **🖌️ Brush** button (left of the white square).  \n"
                        "2. Draw **one digit (0-9)**, big and centered.  \n"
                        "3. Click **Read**. To start again, click the 🗑️ (*Clear canvas*) button at the top.")
            with gr.Row():
                canvas = gr.Sketchpad(label="Draw here", type="numpy", canvas_size=(280, 280),
                                      brush=gr.Brush(default_size=18, colors=["#000000"], color_mode="fixed"),
                                      layers=False, transforms=())
                with gr.Column():
                    probabilities = gr.Label(label="Probabilities", num_top_classes=3)
                    seen = gr.Image(label="What the model sees (28x28)", height=140, width=140)
                    digit_message = gr.Markdown()
            last_drawing = gr.State(None)                   # the 28x28 image of the last drawing read
            gr.Button("Read", variant="primary").click(
                read_drawing, inputs=canvas, outputs=[probabilities, seen, digit_message, last_drawing])

            gr.Markdown("### 🎓 Teach Wakil AI\nWrong answer? Choose the right digit and click **Teach**: "
                        "Wakil AI learns from your drawing right away (mixed with old examples so it doesn't "
                        "forget, and a safety check cancels any update that hurts its MNIST accuracy).")
            with gr.Row():
                correct_digit = gr.Radio([str(d) for d in range(10)], label="The correct digit is")
                with gr.Column():
                    teach_button = gr.Button("Teach Wakil AI", variant="secondary")
                    reset_button = gr.Button("Reset to the original model", size="sm")
            teach_message = gr.Markdown()
            teach_button.click(service.teach, inputs=[last_drawing, correct_digit], outputs=teach_message)
            reset_button.click(service.reset_personal_model, outputs=teach_message)

        with gr.Tab("💬 Write text"):
            gr.Markdown("Type the beginning of a text (English, Shakespeare style). Wakil AI continues it, "
                        "one character at a time.")
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="Beginning", value="ROMEO:\n", lines=3)
                    model_choice = gr.Dropdown(choices=list(service.text_models) or ["(no model)"],
                                               value=next(iter(service.text_models), "(no model)"), label="Model")
                    length = gr.Slider(50, 1000, value=300, step=50, label="Length (characters)")
                    temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.1,
                                            label="Temperature (low = safe and repetitive, high = creative and chaotic)")
                    seed = gr.Number(value=None, label="Seed (optional: same seed = same text)", precision=0)
                    write_button = gr.Button("Write", variant="primary")
                with gr.Column():
                    output_text = gr.Textbox(label="Wakil AI writes", lines=18)
                    text_message = gr.Markdown()
            write_button.click(service.write, inputs=[prompt, model_choice, length, temperature, seed],
                               outputs=[output_text, text_message])

        with gr.Tab("ℹ️ Models"):
            gr.Markdown(service.info())
            gr.Markdown("**Honest note:** these are tiny educational models (about 100,000 parameters, "
                        "trained in minutes on a CPU). Assistants like Claude or ChatGPT are built on the same "
                        "ideas but are millions of times bigger, trained on enormous datasets with huge compute.")
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wakil AI application")
    parser.add_argument("--share", action="store_true", help="create a temporary public link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_app().launch(share=args.share, server_port=args.port)
