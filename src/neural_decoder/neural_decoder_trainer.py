import os
import pickle
import time

from edit_distance import SequenceMatcher
import hydra
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .model import *
from .dataset import SpeechDataset


def getDatasetLoaders(
    datasetName,
    batchSize,
):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)

        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )

    train_ds = SpeechDataset(loadedData["train"], transform=None)
    test_ds = SpeechDataset(loadedData["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData


def build_transition_targets(y, y_len, n_classes):
    """
    y: [B, max_seq_len],  targets in 1..n_classes, padded with 0
    y_len: [B], lengths of each target sequence (excluding padding)
    Returns:
        y_trans: [B, max_trans_len], transition IDs (>=1), padded with 0
        y_trans_len: [B], lengths of each transition sequence (= y_len-1, clipped at >=0)
    """
    B = y.size(0)
    y_len = y_len.clone()

    # transition length L_i = max(y_len_i - 1, 0)
    y_trans_len = torch.clamp(y_len - 1, min=0)

    if y_trans_len.max() == 0:
        # degenerate case (all lengths <=1); just return zeros
        max_trans_len = 1
    else:
        max_trans_len = int(y_trans_len.max().item())

    y_trans = torch.zeros(
        (B, max_trans_len), dtype=torch.long, device=y.device
    )

    for i in range(B):
        Li = int(y_len[i].item())
        if Li <= 1:
            continue
        # true sequence (no padding)
        seq = y[i, :Li]  # [Li]
        # transitions: (p_t, p_{t+1}) → ID = p_t * n_classes + p_{t+1}
        # NOTE: assumes phoneme labels are in [1..n_classes]; 0 is padding
        first = seq[:-1]
        second = seq[1:]
        trans_ids = first * n_classes + second  # [Li-1]
        y_trans[i, :Li - 1] = trans_ids

    return y_trans, y_trans_len

def trainModel(args):
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = "cuda"
    cr_lambda = 0 if "cr_lambda" not in args else args["cr_lambda"]
    transition_loss_weight = 0 if "transitionLossWeight" not in args else args["transitionLossWeight"]

    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    trainLoader, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
    )

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=len(loadedData["train"]),
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
    ).to(device)

    # model = GRUDecoderWithHead(
    #     neural_dim=args["nInputFeatures"],
    #     n_classes=args["nClasses"],
    #     hidden_dim=args["nUnits"],
    #     layer_dim=args["nLayers"],
    #     nDays=len(loadedData["train"]),
    #     dropout=args["dropout"],
    #     device=device,
    #     strideLen=args["strideLen"],
    #     kernelLen=args["kernelLen"],
    #     gaussianSmoothWidth=args["gaussianSmoothWidth"],
    #     bidirectional=args["bidirectional"],
    # ).to(device)
    
    # model = TransformerDecoder(
    #     neural_dim=args["nInputFeatures"],
    #     n_classes=args["nClasses"],
    #     strideLen=args["strideLen"],
    #     kernelLen=args["kernelLen"],
    #     gaussianSmoothWidth=args["gaussianSmoothWidth"],
    #     nDays=len(loadedData["train"]),
    #     n_layers=args["nLayers"],
    #     d_model=args["nUnits"],
    #     n_heads=args["nHeads"],
    #     dropout=args["dropout"],
    # ).to(device)
    
    # model = Hybrid(
    #     neural_dim=args["nInputFeatures"],
    #     n_classes=args["nClasses"],
    #     hidden_dim=args["nUnits"],
    #     n_heads=args["nHeads"],
    #     n_layers=args["nLayers"],
    #     layer_dim=args["nLayers"],
    #     nDays=len(loadedData["train"]),
    #     dropout=args["dropout"],
    #     device=device,
    #     strideLen=args["strideLen"],
    #     kernelLen=args["kernelLen"],
    #     gaussianSmoothWidth=args["gaussianSmoothWidth"],
    #     bidirectional=args["bidirectional"],
    # ).to(device)
    
    # model = GRUTransitionDecoder(
    #     neural_dim=args["nInputFeatures"],
    #     n_classes=args["nClasses"],
    #     hidden_dim=args["nUnits"],
    #     layer_dim=args["nLayers"],
    #     nDays=len(loadedData["train"]),
    #     dropout=args["dropout"],
    #     device=device,
    #     strideLen=args["strideLen"],
    #     kernelLen=args["kernelLen"],
    #     gaussianSmoothWidth=args["gaussianSmoothWidth"],
    #     bidirectional=args["bidirectional"],
    # ).to(device)
    
    # neuralModel1 = GRUDecoder(
    #     neural_dim=args["nInputFeatures"],
    #     n_classes=args["nClasses"],
    #     hidden_dim=args["nUnits1"],
    #     layer_dim=args["nLayers1"],
    #     nDays=len(loadedData["train"]),
    #     dropout=args["dropout"],
    #     device=device,
    #     strideLen=args["strideLen"],
    #     kernelLen=args["kernelLen"],
    #     gaussianSmoothWidth=args["gaussianSmoothWidth1"],
    #     bidirectional=args["bidirectional"],
    # ).to(device)
    
    # neuralModel2 = GRUDecoder(
    #     neural_dim=args["nInputFeatures"],
    #     n_classes=args["nClasses"],
    #     hidden_dim=args["nUnits2"],
    #     layer_dim=args["nLayers2"],
    #     nDays=len(loadedData["train"]),
    #     dropout=args["dropout"],
    #     device=device,
    #     strideLen=args["strideLen"],
    #     kernelLen=args["kernelLen"],
    #     gaussianSmoothWidth=args["gaussianSmoothWidth2"],
    #     bidirectional=args["bidirectional"],
    # ).to(device)
    
    # neuralModel2 = TransformerDecoder(
    #     neural_dim=args["nInputFeatures"],
    #     n_classes=args["nClasses"],
    #     strideLen=args["strideLen"],
    #     kernelLen=args["kernelLen"],
    #     gaussianSmoothWidth=args["gaussianSmoothWidth2"],
    #     nDays=len(loadedData["train"]),
    #     n_layers=args["nLayers2"],
    #     d_model=args["nUnits2"],
    #     n_heads=args["nHeads"],
    #     dropout=args["dropout"],
    # ).to(device)

    # model = EnsembledTransitionWrapper(nn.ModuleList([neuralModel1, neuralModel2]), num_classes=args["nClasses"] + 1).to(device)

    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args["lrStart"],
        betas=(0.9, 0.999),
        eps=0.1,
        weight_decay=args["l2_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args["nBatch"],
        eta_min=args["lrEnd"],
    )
    if "ensembleWeightStart" in args and "ensembleWeightEnd" in args:
        ensemble_weight_scheduler = CosineAnnealing(weight_start=args["ensembleWeightStart"], weight_end=args["ensembleWeightEnd"], total_len=args["nBatch"])
        ensemble_weight = args["ensembleWeightStart"]
    else:
        ensemble_weight = 0.0
    

    # --train--
    testLoss = []
    testCER = []
    startTime = time.time()
    for batch in range(args["nBatch"]):
        if "ensembleWeightStart" in args and "ensembleWeightEnd" in args:
            ensemble_weight = ensemble_weight_scheduler(batch)
        model.train()

        X, y, X_len, y_len, dayIdx = next(iter(trainLoader))
        X, y, X_len, y_len, dayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            dayIdx.to(device),
        )
        adjusted_x_len = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)
        X1 = X.clone()
        # Noise augmentation is faster on GPU
        if args["whiteNoiseSD"] > 0:
            X1 += torch.randn(X.shape, device=device) * args["whiteNoiseSD"]

        if args["constantOffsetSD"] > 0:
            X1 += (
                torch.randn([X.shape[0], 1, X.shape[2]], device=device)
                * args["constantOffsetSD"]
            )
            
        X2 = X1.clone()
        # Noise augmentation is faster on GPU
        if args["whiteNoiseSD"] > 0:
            X2 += torch.randn(X.shape, device=device) * args["whiteNoiseSD"]

        if args["constantOffsetSD"] > 0:
            X2 += (
                torch.randn([X.shape[0], 1, X.shape[2]], device=device)
                * args["constantOffsetSD"]
            )
        

        # Compute prediction error
        
        # Normal Forward pass
        pred1 = model.forward(X1, dayIdx)
        pred2 = model.forward(X2, dayIdx)
        
        # Transition Model forward pass
        # pred1, txLogits1 = model.forward(X1, dayIdx)
        # pred2, txLogits2 = model.forward(X2, dayIdx)
        
        # Ensembled Transition Model forward pass
        # pred1, untrained_preds1 = model.forward(X1, dayIdx)
        # pred2, untrained_preds2 = model.forward(X2, dayIdx)

        # print(pred.shape, y.shape, X_len, y_len)
        # print separately
        # print("pred:", pred.shape)
        # print("y:", y.shape)
        # print("X_len:", X_len)
        # print("y_len:", y_len)
        ctc_loss = loss_ctc(
            torch.permute(pred1.log_softmax(2), [1, 0, 2]),
            y,
            adjusted_x_len,
            y_len,
        ) + loss_ctc(
            torch.permute(pred2.log_softmax(2), [1, 0, 2]),
            y,
            adjusted_x_len,
            y_len,
        )
        ctc_loss = torch.sum(ctc_loss) / 2.0
        log_p1 = pred1.log_softmax(2)  # [B, T, C]
        log_p2 = pred2.log_softmax(2)

        p1 = log_p1.exp()
        p2 = log_p2.exp()

        # Symmetric KL: KL(p1 || p2) + KL(p2 || p1)
        kl_1_2 = (p1 * (log_p1 - log_p2)).sum(dim=-1)  # [B, T]
        kl_2_1 = (p2 * (log_p2 - log_p1)).sum(dim=-1)  # [B, T]
        consistency_loss = (kl_1_2 + kl_2_1).mean()
        
        # Transition loss
        loss_transition = 0.0
        if transition_loss_weight > 0:
                   # Forward pass: two heads
            # --- transition CTC loss ---
            y_trans, y_trans_len = build_transition_targets(
                y, y_len, n_classes=args["nClasses"]
            )
            loss_transition = loss_ctc(
                torch.permute(txLogits1.log_softmax(2), [1, 0, 2]),
                y_trans,
                adjusted_x_len,
                y_trans_len,
            ) + loss_ctc(
                torch.permute(txLogits2.log_softmax(2), [1, 0, 2]),
                y_trans,
                adjusted_x_len,
                y_trans_len,
            )
            loss_transition = torch.sum(loss_transition) / 2.0

        # Ensembled Transition loss
        ensembled_loss_sum = 0.0
        if ensemble_weight > 0:
            for (untrained_pred1, untrained_pred2) in zip(untrained_preds1, untrained_preds2):
                ensembled_loss = loss_ctc(
                    torch.permute(untrained_pred1.log_softmax(2), [1, 0, 2]),
                    y,
                    adjusted_x_len,
                    y_len,
                ) + loss_ctc(
                    torch.permute(untrained_pred2.log_softmax(2), [1, 0, 2]),
                    y,
                    adjusted_x_len,
                    y_len,
                )
                ensembled_loss_sum += torch.sum(ensembled_loss) / (2.0 * len(untrained_preds1))
            ctc_loss += ensemble_weight * ensembled_loss_sum



        loss = ctc_loss + cr_lambda * consistency_loss + transition_loss_weight * loss_transition
        
        # --------------------------------
        
 

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # print(endTime - startTime)

        # Eval
        if batch % 100 == 0:
            with torch.no_grad():
                model.eval()
                allLoss = []
                total_edit_distance = 0
                total_seq_length = 0
                for X, y, X_len, y_len, testDayIdx in testLoader:
                    X, y, X_len, y_len, testDayIdx = (
                        X.to(device),
                        y.to(device),
                        X_len.to(device),
                        y_len.to(device),
                        testDayIdx.to(device),
                    )

                    pred = model.forward(X, testDayIdx)
                    # pred, _ = model.forward(X, testDayIdx)
                    loss = loss_ctc(
                        torch.permute(pred.log_softmax(2), [1, 0, 2]),
                        y,
                        ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
                        y_len,
                    )
                    loss = torch.sum(loss)
                    allLoss.append(loss.cpu().detach().numpy())

                    adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(
                        torch.int32
                    )
                    for iterIdx in range(pred.shape[0]):
                        decodedSeq = torch.argmax(
                            torch.tensor(pred[iterIdx, 0 : adjustedLens[iterIdx], :]),
                            dim=-1,
                        )  # [num_seq,]
                        decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                        decodedSeq = decodedSeq.cpu().detach().numpy()
                        decodedSeq = np.array([i for i in decodedSeq if i != 0])

                        trueSeq = np.array(
                            y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
                        )

                        matcher = SequenceMatcher(
                            a=trueSeq.tolist(), b=decodedSeq.tolist()
                        )
                        total_edit_distance += matcher.distance()
                        total_seq_length += len(trueSeq)

                avgDayLoss = np.sum(allLoss) / len(testLoader)
                cer = total_edit_distance / total_seq_length

                endTime = time.time()
                print(
                    f"batch {batch}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}, time/batch: {(endTime - startTime)/100:>7.3f}"
                )
                startTime = time.time()

            if len(testCER) > 0 and cer < np.min(testCER):
                torch.save(model.state_dict(), args["outputDir"] + "/modelWeights")
            testLoss.append(avgDayLoss)
            testCER.append(cer)

            tStats = {}
            tStats["testLoss"] = np.array(testLoss)
            tStats["testCER"] = np.array(testCER)

            with open(args["outputDir"] + "/trainingStats", "wb") as file:
                pickle.dump(tStats, file)


def loadModel(modelDir, nInputLayers=24, device="cuda"):
    modelWeightPath = modelDir + "/modelWeights"
    with open(modelDir + "/args", "rb") as handle:
        args = pickle.load(handle)

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=nInputLayers,
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
    ).to(device)

    model.load_state_dict(torch.load(modelWeightPath, map_location=device))
    return model


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg):
    cfg.outputDir = os.getcwd()
    trainModel(cfg)

if __name__ == "__main__":
    main()