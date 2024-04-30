import argparse
from transformers import AutoTokenizer

from utils import *
from train_utils import *
from model import *


def main(args):
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    train_loader = get_data_loader(
        'data/in_domain_train.tsv', args.batch_size, tokenizer)
    val_loader = get_data_loader(
        'data/in_domain_dev.tsv', args.batch_size, tokenizer, shuffle=False)
    
    print("MODE: ", args.mode)
    

    if args.mode == "gen":
        model = GPT(args.gpt_variant, is_gen=True).to(args.device)
        model.eval()

        # TODO: You can add your super creative prompt here
        prompt = "My name is Inigo Montoya. You killed my father. Prepare to die."

        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(args.device)
        output = model.generate(input_ids, max_new_tokens=args.max_new_tokens)
        print("", tokenizer.decode(output[0]), sep="\n")

    elif args.mode == "LoRA":    
        model = GPT(args.gpt_variant, LoRA_rank=args.LoRA_rank).to(args.device)

        # gpt_model = GPT(model_type='gpt2', is_gen=False)  # Initialize with appropriate type
        # training_losses = train(gpt_model, train_dataset, device='cuda')
        # validation_accuracies = [evaluate(gpt_model, val_dataset, device='cuda') for _ in range(epochs)]
        # plot_metrics(training_losses, validation_accuracies)
        training_losses = []
        training_accuracies = []
        validation_losses = []
        validation_accuracies = []
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        loss_fn = torch.nn.CrossEntropyLoss()


        for epoch in range(args.epochs):
            loss, acc = train(model, train_loader,optimizer,loss_fn , args.device)
            print(f"Epoch {epoch+1}: Train Loss {loss}, Train Accuracy {acc}")
            training_losses.append(loss)
            training_accuracies.append(acc)
            loss, acc = evaluate(model, val_loader,loss_fn, args.device)
            print(f"Epoch {epoch+1}: Validation Loss {loss}, Validation Accuracy {acc}")
            validation_losses.append(loss)
            validation_accuracies.append(acc)
        
        plot_metrics(training_losses,validation_losses, training_accuracies, validation_accuracies)


        # training_losses, training_accuracies = train(model, train_loader, device, epochs=5)
        # validation_accuracies = [evaluate(model, val_loader, device) for _ in range(5)]
        # plot_metrics(training_losses, validation_accuracies)

        # TODO: Implement the training loop (fill the train and evaluate functions in train_utils.py)
        # TODO: Also plot the training losses and metrics

        model.save_trainable_params(args.model_path)
        
    elif args.mode == "distil":
        teacher_model = GPT(args.gpt_variant, LoRA_rank=args.LoRA_rank).to(args.device)
        # teacher_model.load_trainable_params(args.model_path)
        teacher_model.eval()

        student_model = DistilRNN().to(args.device)  # TODO: Implement the student model class

        training_losses = []
        training_accuracies = []
        validation_losses = []
        validation_accuracies = []
        optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.lr)
        loss_fn = torch.nn.CrossEntropyLoss()


        for epoch in range(args.epochs):
            loss, acc = train3(teacher_model, student_model, train_loader,optimizer,loss_fn, 2, args.device, 0.25, 0.75)
            print(f"Epoch {epoch+1}: Train Loss {loss}, Train Accuracy {acc}")
            training_losses.append(loss)
            training_accuracies.append(acc)
            loss, acc = evaluate(student_model, val_loader,loss_fn, args.device)
            print(f"Epoch {epoch+1}: Validation Loss {loss}, Validation Accuracy {acc}")
            validation_losses.append(loss)
            validation_accuracies.append(acc)
        
        plot_metrics(training_losses,validation_losses, training_accuracies, validation_accuracies)


        # TODO: Implement the training loop (fill the train and evaluate functions in train_utils.py)
        # HINT: You can use an additional parameter in train function to differentiate LoRA and distillation training, no changes in evaluate function required.
        #raise NotImplementedError
    elif args.mode == "rnn":
        # for data in train_loader:
        #     print(data)
        #     break
        # print(input_size)
        # print(data[0].shape)
        # input_size = 768
        # hidden_size = 2*768
        # num_layers = 10
        # output_size = 2
        model = DistilRNN().to(args.device)
        print("VOILA")

        training_losses = []
        training_accuracies = []
        validation_losses = []
        validation_accuracies = []
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        loss_fn = torch.nn.CrossEntropyLoss()


        for epoch in range(args.epochs):
            loss, acc = train2(model, train_loader,optimizer,loss_fn , args.device)
            print(f"Epoch {epoch+1}: Train Loss {loss}, Train Accuracy {acc}")
            training_losses.append(loss)
            training_accuracies.append(acc)
            loss, acc = evaluate(model, val_loader,loss_fn, args.device)
            print(f"Epoch {epoch+1}: Validation Loss {loss}, Validation Accuracy {acc}")
            validation_losses.append(loss)
            validation_accuracies.append(acc)
        
        plot_metrics(training_losses,validation_losses, training_accuracies, validation_accuracies)
        # TODO: Implement the training loop (fill the train and evaluate functions in train_utils.py)
        #raise NotImplementedError
    else:
        print("Invalid mode")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assignment 2")
    parser.add_argument("mode", type=str, choices=["gen", "LoRA", "distil", "rnn"], help="Mode to run the program in")
    parser.add_argument("sr_no", type=int, help="5 digit SR number")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--gpt_variant", type=str, default="gpt2", choices=["gpt2", "gpt2-medium"], help="Model to use")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--model_path", type=str, default="models/LoRA.pth", help="Path to save the model")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--LoRA_rank", type=int, default=4, help="Low rank matrix bottleneck")
    # TODO: Add more arguments as needed
    
    args = parser.parse_args()
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and args.gpu_id >= 0 else\
        "mps" if torch.backends.mps.is_available() else "cpu")
    
    seed_everything(args.sr_no)

    main(args)
