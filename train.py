import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="densenet121",
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,                      # model output channels (number of classes in your dataset)
)

for images, gt_masks in dataset:

    predicted_mask = model(images)
    loss = loss_fn(predicted_mask, gt_masks)

    loss.backward()
    optimizer.step()