from flask import Flask, request, render_template
from factory import ModelFactory

app = Flask(__name__)

# Khởi tạo model factory
factory = ModelFactory(
    tokenizer_path="./tokenizer/tokenizer_word2vec.pkl",
    embedding_path="./embedding/wiki.vi.vec"
)

# Cache các model đã được load
loaded_models = {}

@app.route("/", methods=["GET", "POST"])
def index():
    user_message = ""
    bot_reply = ""
    selected_model = "lstm_beam"  # Model mặc định

    if request.method == "POST":
        user_message = request.form.get("message", "").strip()
        selected_model = request.form.get("model", "lstm")

        if user_message:
            try:
                if selected_model not in loaded_models:
                    loaded_models[selected_model] = factory.get_model(selected_model)
                model = loaded_models[selected_model]
                bot_reply = model.predict(user_message)
            except Exception as e:
                bot_reply = f"Lỗi: {str(e)}"

    return render_template("index.html",
                           user_message=user_message,
                           bot_reply=bot_reply,
                           selected_model=selected_model)

if __name__ == "__main__":
    app.run(debug=True)
