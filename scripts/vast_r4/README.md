# r4 on Vast.ai (2026-11-02)

Fine-tunes r3 on floodgate records (2019-2026, both players rated 3800+, or 3500+ when that gives
fewer than 30M positions), then plays r4 against r3 (800 playouts, SPRT) on the same machine.

- `prep.sh`: downloads the floodgate archives and converts them with utils/csa_to_hcpe.py.
- `pipeline.sh`: waits for prep, trains (resume from r3h.pth = utils/onnx_to_checkpoint.py on r3.onnx, fp16), exports r4.onnx, runs the match, writes /workspace/out/DONE.
- `deadline.sh`: hard cap; stops the instance 16 hours after start, whatever happens.
- `watch.sh` (workstation): waits for DONE, downloads out/ to model/r4 (checksummed), then destroys the instance.

Billing safety: the instance stops itself (Vast API with its own CONTAINER_API_KEY) on a training failure,
3 hours after DONE if nobody fetched the results, and at the 16-hour deadline.
