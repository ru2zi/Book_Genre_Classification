from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup


def get_scheduler(name, optimizer, train_loader, n_epochs, num_warmup_steps_rate, num_cycles=0):
    # print('start_ get_scheduler')

    num_train_optimization_steps = int(len(train_loader) * n_epochs)
    num_warmup_steps = int(num_train_optimization_steps * num_warmup_steps_rate)
    if name == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                  num_warmup_steps=num_warmup_steps,
                                                  num_training_steps=num_train_optimization_steps)
    if name == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                  num_warmup_steps=num_warmup_steps, 
                                                  num_training_steps=num_train_optimization_steps,
                                                  num_cycles=num_cycles)
    # print('end_ get_scheduler')

    return scheduler