import torch


def update_teacher_model(teacher, student, alpha):
    teacher_state_dict = teacher.state_dict()
    student_state_dict = student.state_dict()

    for key in teacher_state_dict.keys():
        dtype = teacher_state_dict[key].dtype
        try:
            teacher_state_dict[key] = (
                alpha * teacher_state_dict[key] + (1 - alpha) * student_state_dict[key]
            )
        # when using DataParallel
        except KeyError:
            teacher_state_dict[key] = (
                alpha * teacher_state_dict[key]
                + (1 - alpha) * student.module.state_dict()[key]
            )
        teacher_state_dict[key].to(dtype)

    teacher.load_state_dict(teacher_state_dict)

    return teacher


def average_model_params(model_path, epochs, filename="separator"):
    filename = filename + ".pth"

    state_dict = torch.load(model_path / ("epoch" + str(epochs[0])) / filename)
    for i, epoch in enumerate(epochs[1:]):
        tmp_state_dict = torch.load(model_path / ("epoch" + str(epoch)) / filename)
        for key in state_dict.keys():
            state_dict[key] += tmp_state_dict[key]

    for key in state_dict.keys():
        dtype = state_dict[key].dtype
        state_dict[key] = (state_dict[key] / len(epochs)).to(dtype)

    return state_dict
