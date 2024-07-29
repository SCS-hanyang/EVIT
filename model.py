

model = ViT(in_channels=3, patch_size=32, emb_size=96, img_size=32, depth = 2, n_classes=10)  # 120 classes in Stanford Dogs dataset

# Define device (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model to device
model.to(device)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

PATH = './weights/'

# Training loop
num_epochs = 3
total = len(trainloader)
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    st = time.time()

    for i, data in enumerate(trainloader, 1):

        images, label = data[0].to(device), data[1].to(device)
        print("\rlearning... %.2f%%"%(i / total * 100), end = '')

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass

        outputs = model(images)


        # Calculate loss
        loss = criterion(outputs, label)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    # Print epoch statistics
    epoch_loss = running_loss / len(trainloader)
    print(f'\nEpoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    correct = 0

    for t_img, t_lab in testloader:
        t_out = model(t_img)
        pred = t_out.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct += pred.eq(t_lab.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(testset)
    print('\n Accuracy: ({%.2f}%%)\n'%(accuracy))

    if not os.path.isdir(PATH) : os.mkdir(PATH)
    ed = time.time()
    epoch_time = ed - st
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'time':epoch_time,
        'epoch':epoch,
        'loss':epoch_loss,
        'accuracy':accuracy,
    }, PATH + f'{epoch + 1}.tar')

print('Training finished!')
