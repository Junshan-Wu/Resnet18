import os

import torch
import torch.nn.functional as F

import DataLoader
import parameters


def _name_safe(value):
    return str(value).replace('.', 'p').replace('-', 'm')


def _build_run_tag(params):
    return (
        f"fullrelay_lr_{_name_safe(params.learning_rate)}_"
        f"bs_{params.batch_size}_"
        f"act_{params.activation}_"
        f"cutout_{params.use_cutout}_"
        f"valid_{_name_safe(params.valid_size)}"
    )


def _set_optimizer_lr(optimizer, learning_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


def _save_history(history_dir, run_tag, loss_history, success_history, train_step_losses):
    os.makedirs(history_dir, exist_ok=True)

    torch.save(loss_history, os.path.join(history_dir, f"{run_tag}_loss_history.pt"))
    torch.save(success_history, os.path.join(history_dir, f"{run_tag}_success_history.pt"))
    torch.save(train_step_losses, os.path.join(history_dir, f"{run_tag}_train_step_losses.pt"))


def _save_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    next_epoch,
    global_step,
    current_lr,
    num_epoch,
    best_train_loss,
    best_model_path,
    loss_history,
    success_history,
    train_step_losses,
):
    checkpoint = {
        'next_epoch': next_epoch,
        'global_step': global_step,
        'current_lr': current_lr,
        'num_epoch': num_epoch,
        'best_train_loss': best_train_loss,
        'best_model_path': best_model_path,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': loss_history,
        'success_history': success_history,
        'train_step_losses': train_step_losses,
    }
    torch.save(checkpoint, checkpoint_path)


def _load_checkpoint(checkpoint_path, model, optimizer, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return (
        checkpoint['next_epoch'],
        checkpoint['global_step'],
        checkpoint['current_lr'],
        checkpoint['best_train_loss'],
        checkpoint['best_model_path'],
        checkpoint['loss_history'],
        checkpoint['success_history'],
        checkpoint['train_step_losses'],
    )


def _prompt_new_learning_rate(current_lr):
    while True:
        user_input = input(
            f"\n请输入新的学习率（直接回车继续使用 {current_lr}；输入 q 退出训练）："
        ).strip()

        if user_input == '':
            return current_lr
        if user_input.lower() in {'q', 'quit', 'exit'}:
            return None

        try:
            new_lr = float(user_input)
            if new_lr <= 0:
                print("学习率必须大于 0，请重新输入。")
                continue
            return new_lr
        except ValueError:
            print("输入无效，请输入一个正数（例如 1e-4）。")


def train_full_relay(model, num_epoch):
    """
    手动中继训练：
    1. 正常训练并打印 loss/accuracy。
    2. 在训练中按 Ctrl+C 会自动保存断点与历史。
    3. 然后可手动输入新学习率，自动从断点加载并继续训练。
    4. 不使用学习率调度器。
    """
    print("Start relay training (manual LR adjustment, no scheduler)")

    params = parameters.get_parameters()
    device = params.device if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.train()

    trainset = DataLoader.Full_train_data_Loader()

    save_dir = params.model_save_dir
    os.makedirs(save_dir, exist_ok=True)

    history_dir = os.path.join('results', 'data', 'relay_history')
    os.makedirs(history_dir, exist_ok=True)

    run_tag = _build_run_tag(params)
    checkpoint_path = os.path.join(save_dir, f"{run_tag}_checkpoint.pth")
    best_model_path = os.path.join(save_dir, f"{run_tag}_best.pth")

    current_lr = params.learning_rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=current_lr)

    start_epoch = 0
    global_step = 0
    loss_history = []
    success_history = []
    train_step_losses = []
    best_train_loss = float('inf')

    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint: {checkpoint_path}")
        choice = input("检测到断点，是否恢复训练？[Y/n]: ").strip().lower()
        if choice in {'', 'y', 'yes'}:
            (
                start_epoch,
                global_step,
                current_lr,
                best_train_loss,
                best_model_path,
                loss_history,
                success_history,
                train_step_losses,
            ) = _load_checkpoint(checkpoint_path, model, optimizer, device)
            print(
                f"Resumed from checkpoint. Next epoch: {start_epoch}/{num_epoch}, "
                f"current lr: {current_lr}"
            )
        else:
            print("Ignore existing checkpoint and start a new run.")

    _set_optimizer_lr(optimizer, current_lr)

    epoch = start_epoch
    while epoch < num_epoch:
        try:
            model.train()
            correct_preds_all = 0
            train_num_all = 0
            total_loss = 0.0

            for step, (images, labels) in enumerate(trainset):
                global_step += 1

                images = images.to(device)
                labels = labels.to(device)
                batch_size = labels.size(0)

                optimizer.zero_grad()
                preds = model(images)
                loss = F.cross_entropy(preds, labels)
                loss.backward()
                optimizer.step()

                loss_value = loss.item()
                total_loss += loss_value * batch_size
                train_step_losses.append(loss_value)

                with torch.no_grad():
                    _, label_preds = preds.max(dim=1)
                    correct_preds = torch.sum(labels == label_preds)
                    success_rate = correct_preds / batch_size
                    correct_preds_all += correct_preds
                    train_num_all += batch_size

                if step % 10 == 0:
                    print(
                        f"epoch: {epoch} train step: {step} "
                        f"accuracy: {success_rate:.3f}, loss: {loss_value:.3f}, lr: {current_lr:.6g}"
                    )

            avg_loss = total_loss / train_num_all
            success_rate_epoch = (correct_preds_all / train_num_all).item()

            loss_history.append(avg_loss)
            success_history.append(success_rate_epoch)

            print(
                f"epoch {epoch} training finish! "
                f"accuracy: {success_rate_epoch:.4f}, average loss: {avg_loss:.6f}, lr: {current_lr:.6g}"
            )

            if avg_loss < best_train_loss:
                best_train_loss = avg_loss
                torch.save(model.state_dict(), best_model_path)
                print(
                    f"Best model updated: {best_model_path}, "
                    f"best training loss: {best_train_loss:.6f}"
                )

            next_epoch = epoch + 1
            _save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                next_epoch=next_epoch,
                global_step=global_step,
                current_lr=current_lr,
                num_epoch=num_epoch,
                best_train_loss=best_train_loss,
                best_model_path=best_model_path,
                loss_history=loss_history,
                success_history=success_history,
                train_step_losses=train_step_losses,
            )
            _save_history(history_dir, run_tag, loss_history, success_history, train_step_losses)

            epoch = next_epoch

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt detected. Saving checkpoint before relay...")

            _save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                next_epoch=epoch,
                global_step=global_step,
                current_lr=current_lr,
                num_epoch=num_epoch,
                best_train_loss=best_train_loss,
                best_model_path=best_model_path,
                loss_history=loss_history,
                success_history=success_history,
                train_step_losses=train_step_losses,
            )
            _save_history(history_dir, run_tag, loss_history, success_history, train_step_losses)
            print(f"Checkpoint saved to: {checkpoint_path}")

            new_lr = _prompt_new_learning_rate(current_lr)
            if new_lr is None:
                print("Training stopped by user. Current checkpoint and history have been saved.")
                break

            (
                start_epoch,
                global_step,
                _,
                best_train_loss,
                best_model_path,
                loss_history,
                success_history,
                train_step_losses,
            ) = _load_checkpoint(checkpoint_path, model, optimizer, device)

            current_lr = new_lr
            _set_optimizer_lr(optimizer, current_lr)
            epoch = start_epoch

            print(
                f"Relay continue from epoch {epoch}/{num_epoch} with new learning rate {current_lr}"
            )

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model weights from {best_model_path}")

    _save_history(history_dir, run_tag, loss_history, success_history, train_step_losses)

    return model, loss_history, success_history, train_step_losses
