import pytest
import os
import subprocess


def file_exists(directory_path, file_name):
    return os.path.isfile(os.path.join(directory_path, file_name))


@pytest.fixture(params=["2", "3"])
def zero_stage(request):
    return str(request.param)


@pytest.fixture(params=["true", "false"])
def hybrid_engine(request):
    return str(request.param)


@pytest.fixture(params=["true", "false"])
def offload(request):
    return str(request.param)


@pytest.fixture(params=["true", "false"])
def lora(request):
    return str(request.param)


@pytest.fixture
def params(zero_stage, hybrid_engine, offload, lora):
    #model = "facebook/opt-125m"
    actor_model = "AdamG012/chat-opt-1.3b-sft-deepspeed"
    critic_model = "AdamG012/chat-opt-350m-reward-deepspeed"
    output_path = "z" + zero_stage + "_he_" + hybrid_engine + "_offload_" + offload + "_lora_" + lora

    return [
        actor_model,
        critic_model,
        zero_stage,
        zero_stage,
        hybrid_engine,
        offload,
        lora,
        output_path,
    ]


#@pytest.mark.parametrize('zero_stage', [2, 3])
#@pytest.mark.parametrize('hybrid_engine', [True, False])
#@pytest.mark.parametrize('offload', [True, False])
#@pytest.mark.parametrize('lora', [True, False])
def test_ds_chat(params):
    # TODO (lekurile): Add test-only params to main.py (test_enable, test_breakpoint)
    if params[3] == "2" and params[4] == "true" and params[
            5] == "true" and params[6] == "false":
        pytest.skip(
            "The combination of [actor_zero_stage==2, critic_zero_stage==2, enable_hybrid_engine=True, offload=True, lora=False] is currently unsupported due to training instability!"
        )

    if params[3] == "3" and params[4] == "true" and params[
            5] == "true" and params[6] == "true":
        pytest.skip(
            "The combination of [actor_zero_stage==3, critic_zero_stage==3, enable_hybrid_engine=True, offload=True, lora=True] is currently unsupported due to training instability!"
        )

    wd = os.getcwd()
    os.chdir("../step3_rlhf_finetuning")

    sweep_script = "training_scripts/single_node/sweep/run_single.sh"

    #import pdb; pdb.set_trace()
    # Run bash script
    cmd = ["bash", sweep_script] + params

    result = subprocess.run(cmd)
    result.check_returncode()
    #import pdb; pdb.set_trace()

    # Assert Model files exist
    assert file_exists(f"{params[-1]}/actor/", "pytorch_model.bin"
                       ), "Actor model was not saved during step 3 training."
    assert file_exists(f"{params[-1]}/critic/", "pytorch_model.bin"
                       ), "Critic model was not saved during step 3 training."
