from utils import *
#from train_model import *
from test_model import *

    
def GUI(model,words,classes):

    root = tk.Tk()
    root.title("Medical ChatBot")
    root.geometry("700x700")
    root.configure(bg="lightblue")

    input_label = tk.Label(root, text="Enter your message:", bg="lightblue", fg="darkblue", font=("Bold", 12))
    input_label.pack(pady=8)
    input_text = tk.Text(root, height=1, width=50, bg="white")
    input_text.pack(pady=8, padx=6)
    input_text.bind("<Return>", lambda *args: classify_and_respond(input_text, model, words, classes, response_text,root))

    response_label = tk.Label(root, text="ChatBot Conversation", bg="lightblue", fg="darkblue", font=("Bold", 12))
    response_label.pack(pady=8)
    response_text = scrolledtext.ScrolledText(root, height=20, width=100, bg="white")
    response_text.pack(pady=10, padx=10)
    # response_text.destroy()

    style = ttk.Style()
    style.configure("TButton", foreground="black", background="blue", font=("Bold", 15))
    # send_button = ttk.Button(root, text="Send", command=classify_and_respond(input_text,model,words,classes,response_text), style="TButton")
    send_button = ttk.Button(root, text="Send", command=lambda: classify_and_respond(input_text, model, words, classes, response_text,root), style="TButton")
    send_button.pack(pady=15)

    root.mainloop()

    

def main():
    model,words,classes= test()
    GUI(model,words,classes)
        
if __name__ == "__main__":
    # Call the function to create the chatbot interface
    main()
