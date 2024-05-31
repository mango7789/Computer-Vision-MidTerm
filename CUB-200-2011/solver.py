import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import preprocess_data
from model import CUB_ResNet_18
from tqdm import tqdm

def seed_everything(seed: int=None):
    """
    Set the random seed for the whole neural network.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_data_model_criterion(data_dir: str, pretrain: bool=True) -> tuple:
    """
    Get the DataLoader, model and loss criterion.
    """
    # load the dataset
    train_loader, test_loader = preprocess_data(data_dir)

    # get the pretrained model
    model = CUB_ResNet_18(pretrain=pretrain)

    # define loss function
    criterion = nn.CrossEntropyLoss()
    
    return train_loader, test_loader, model, criterion

def calculate_topk_correct(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)) -> list[int]:
    """
    Computes the top-k correct samples for the specified values of k.

    Args:
    - output (torch.Tensor): The model predictions with shape (batch_size, num_classes).
    - target (torch.Tensor): The true labels with shape (batch_size, ).
    - topk (tuple): A tuple of integers specifying the top-k values to compute.

    Returns:
    - List of top-k correct samples for each value in topk.
    """
    maxk = max(topk)

    # get the indices of the top k predictions
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.item())
    return res

def train_resnet_with_cub(
    data_dir: str,
    num_epochs: list[int], 
    fine_tuning_lr: float=0.0001, 
    output_lr: float=0.001, 
    pretrain: bool=True, 
    save: bool=False,
    **kwargs: dict
) -> list[float]:
    """
    Train the modified ResNet-18 model using the CUB-200-2011 dataset and return the best accuracy.
    Some hyper-parameters can be modified here.
    
    Args:
    - data_dir: The stored directory of the dataset.
    - num_epochs: A list of number of training epochs.
    - fine_tuning_lr: Learning rate of the parameters outside the output layer, default is 0.0001.
    - output_lr: Learning rate of the parameters inside the output layer, default is 0.001.
    - pretrain: Boolean, whether the ResNet-18 model is pretrained or not. Default is True.
    - save: Boolean, whether the parameters of the best model will be save. Default is False.
    
    Return:
    - best_acc: The best validation accuracy list during the training process.
    """
    # set the random seed
    seed_everything(kwargs.pop('seed', 42))
    
    # get the dataset, model and loss criterion
    train_loader, test_loader, model, criterion = get_data_model_criterion(data_dir, pretrain)
    
    # move the model to CUDA (GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # get the parameters of the model expect the last layer
    former_params = [p for name, p in model.resnet18.named_parameters() if 'fc' not in name]
    
    # pop the hyper-parameters from the kwargs dict
    momentum = kwargs.pop('momentum', 0.9)
    weight_decay = kwargs.pop('weight_decay', 1e-4)
        
    # define optimizer
    optimizer = optim.SGD([
                {'params': former_params, 'lr': fine_tuning_lr, 'weight_decay': weight_decay},
                {'params': model.resnet18.fc.parameters(), 'lr': output_lr, 'weight_decay': weight_decay}
            ], momentum=momentum
        )
    
    # scheduler step size and gamma
    step_size = kwargs.pop('step', 30)
    gamma = kwargs.pop('gamma', 0.1)

    # custom step scheduler
    def custom_step_scheduler(optimizer: optim, epoch: int, step_size: int, gamma: float):
        """
        Decay the learning rate of the second parameter group by gamma every step_size epochs.
        """
        if epoch % step_size == 0 and epoch > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= gamma
    
    # init the tensorboard
    tensorboard_name = "/kaggle/working/Fine_Tuning_With_Pretrain"
    if len(num_epochs) != 1:
        tensorboard_name = "/kaggle/working/Full_Train"
    if not pretrain:
        tensorboard_name = '/kaggle/working/Random_Init'
    writer = SummaryWriter(tensorboard_name, comment="-{}-{}".format(fine_tuning_lr, output_lr))
        
    # best accuracy
    best_acc = 0.0
    store_best_acc, count = [0 for _ in range(len(num_epochs))], 0
    max_num_epoch = max(num_epochs)

    print("=" * 70)
    print("Training with configuration ({:>7.5f}, {:>7.5f})".format(fine_tuning_lr, output_lr))
    
    # iterate
    for epoch in range(max_num_epoch):
        # train
        model.train()
        samples = 0
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            samples += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)
        
        # learning rate decay
        custom_step_scheduler(optimizer, epoch, step_size, gamma)
        
        train_loss = running_loss / samples
        print("[Epoch {:>2} / {:>2}], Training loss is {:>8.6f}".format(epoch + 1, max_num_epoch, train_loss))

        # test
        model.eval()
        correct_top1 = 0
        correct_top5 = 0
        samples = 0
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                top1, top5 = calculate_topk_correct(outputs, labels, topk=(1, 5))
                correct_top1 += top1
                correct_top5 += top5
                samples += labels.size(0)
                
                running_loss += criterion(outputs, labels).item() * inputs.size(0)

        # add loss and accuracy to tensorboard
        test_loss = running_loss / samples
        writer.add_scalars('Loss', {'Train': train_loss, 'Valid': test_loss}, epoch + 1)
        accuracy_top1 = correct_top1 / samples
        accuracy_top5 = correct_top5 / samples
        writer.add_scalars('Valid Accuracy', {'Top1': accuracy_top1, 'Top5': accuracy_top5}, epoch + 1)
        
        print("[Epoch {:>2} / {:>2}], Validation loss is {:>8.6f}, Top-5 accuracy is {:>8.6f}, Top-1 accuracy is {:>8.6f}".format(
            epoch + 1, max_num_epoch, test_loss, accuracy_top5, accuracy_top1
        ))
        
        # update the best accuracy and save the model if it improves
        if accuracy_top1 > best_acc:
            best_acc = accuracy_top1
            if save:
                if not os.path.exists('./Output'):
                    os.mkdir('./Output')
                torch.save(model.state_dict(), os.path.join('Output', 'resnet18_cub.pth'))
            
        if epoch + 1 == num_epochs[count]:
            store_best_acc[count] = best_acc
            count += 1

    # close the tensorboard
    writer.close()
    
    return store_best_acc

def test_resnet_with_cub(data_dir: str, path: str):
    """
    Test the trained model on the CUB dataset.
    
    Args:
    - data_dir: The stored directory of the dataset.
    - path: Path to the .pth file. 
    """
    # get the dataset, model and loss criterion
    train_loader, test_loader, model, _ = get_data_model_criterion(data_dir)
    bird_species = [
        "Black_footed_Albatross", "Laysan_Albatross", "Sooty_Albatross", "Groove_billed_Ani", "Crested_Auklet", 
        "Least_Auklet", "Parakeet_Auklet", "Rhinoceros_Auklet", "Brewer_Blackbird", "Red_winged_Blackbird", 
        "Rusty_Blackbird", "Yellow_headed_Blackbird", "Bobolink", "Indigo_Bunting", "Lazuli_Bunting", 
        "Painted_Bunting", "Cardinal", "Spotted_Catbird", "Gray_Catbird", "Yellow_breasted_Chat", 
        "Eastern_Towhee", "Chuck_will_Widow", "Brandt_Cormorant", "Red_faced_Cormorant", "Pelagic_Cormorant", 
        "Bronzed_Cowbird", "Shiny_Cowbird", "Brown_Creeper", "American_Crow", "Fish_Crow", 
        "Black_billed_Cuckoo", "Mangrove_Cuckoo", "Yellow_billed_Cuckoo", "Gray_crowned_Rosy_Finch", "Purple_Finch", 
        "Northern_Flicker", "Acadian_Flycatcher", "Great_Crested_Flycatcher", "Least_Flycatcher", "Olive_sided_Flycatcher", 
        "Scissor_tailed_Flycatcher", "Vermilion_Flycatcher", "Yellow_bellied_Flycatcher", "Frigatebird", "Northern_Fulmar", 
        "Gadwall", "American_Goldfinch", "European_Goldfinch", "Boat_tailed_Grackle", "Eared_Grebe", 
        "Horned_Grebe", "Pied_billed_Grebe", "Western_Grebe", "Blue_Grosbeak", "Evening_Grosbeak", 
        "Pine_Grosbeak", "Rose_breasted_Grosbeak", "Pigeon_Guillemot", "California_Gull", "Glaucous_winged_Gull", 
        "Heermann_Gull", "Herring_Gull", "Ivory_Gull", "Ring_billed_Gull", "Slaty_backed_Gull", 
        "Western_Gull", "Anna_Hummingbird", "Ruby_throated_Hummingbird", "Rufous_Hummingbird", "Green_Violetear", 
        "Long_tailed_Jaeger", "Pomarine_Jaeger", "Blue_Jay", "Florida_Jay", "Green_Jay", 
        "Dark_eyed_Junco", "Tropical_Kingbird", "Gray_Kingbird", "Belted_Kingfisher", "Green_Kingfisher", 
        "Pied_Kingfisher", "Ringed_Kingfisher", "White_breasted_Kingfisher", "Red_legged_Kittiwake", "Horned_Lark", 
        "Pacific_Loon", "Mallard", "Western_Meadowlark", "Hooded_Merganser", "Red_breasted_Merganser", 
        "Mockingbird", "Nighthawk", "Clark_Nutcracker", "White_breasted_Nuthatch", "Baltimore_Oriole", 
        "Hooded_Oriole", "Orchard_Oriole", "Scott_Oriole", "Ovenbird", "Brown_Pelican", 
        "White_Pelican", "Western_Wood_Pewee", "Sayornis", "American_Pipit", "Whip_poor_Will", 
        "Horned_Puffin", "Common_Raven", "White_necked_Raven", "American_Redstart", "Geococcyx", 
        "Loggerhead_Shrike", "Great_Grey_Shrike", "Baird_Sparrow", "Black_throated_Sparrow", "Brewer_Sparrow", 
        "Chipping_Sparrow", "Clay_colored_Sparrow", "House_Sparrow", "Field_Sparrow", "Fox_Sparrow", 
        "Grasshopper_Sparrow", "Harris_Sparrow", "Henslow_Sparrow", "Le_Conte_Sparrow", "Lincoln_Sparrow", 
        "Nelson_Sharp_tailed_Sparrow", "Savannah_Sparrow", "Seaside_Sparrow", "Song_Sparrow", "Tree_Sparrow", 
        "Vesper_Sparrow", "White_crowned_Sparrow", "White_throated_Sparrow", "Cape_Glossy_Starling", "Bank_Swallow", 
        "Barn_Swallow", "Cliff_Swallow", "Tree_Swallow", "Scarlet_Tanager", "Summer_Tanager", 
        "Artic_Tern", "Black_Tern", "Caspian_Tern", "Common_Tern", "Elegant_Tern", 
        "Forsters_Tern", "Least_Tern", "Green_tailed_Towhee", "Brown_Thrasher", "Sage_Thrasher", 
        "Black_capped_Vireo", "Blue_headed_Vireo", "Philadelphia_Vireo", "Red_eyed_Vireo", "Warbling_Vireo", 
        "White_eyed_Vireo", "Yellow_throated_Vireo", "Bay_breasted_Warbler", "Black_and_white_Warbler", "Black_throated_Blue_Warbler", 
        "Blue_winged_Warbler", "Canada_Warbler", "Cape_May_Warbler", "Cerulean_Warbler", "Chestnut_sided_Warbler", 
        "Golden_winged_Warbler", "Hooded_Warbler", "Kentucky_Warbler", "Magnolia_Warbler", "Mourning_Warbler", 
        "Myrtle_Warbler", "Nashville_Warbler", "Orange_crowned_Warbler", "Palm_Warbler", "Pine_Warbler", 
        "Prairie_Warbler", "Prothonotary_Warbler", "Swainson_Warbler", "Tennessee_Warbler", "Wilson_Warbler", 
        "Worm_eating_Warbler", "Yellow_Warbler", "Northern_Waterthrush", "Louisiana_Waterthrush", "Bohemian_Waxwing", 
        "Cedar_Waxwing", "American_Three_toed_Woodpecker", "Pileated_Woodpecker", "Red_bellied_Woodpecker", "Red_cockaded_Woodpecker", 
        "Red_headed_Woodpecker", "Downy_Woodpecker", "Bewick_Wren", "Cactus_Wren", "Carolina_Wren", 
        "House_Wren", "Marsh_Wren", "Rock_Wren", "Winter_Wren", "Common_Yellowthroat"
    ]
    
    # move the model to CUDA (GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # load the trained model
    trained_state_dict = torch.load(path, map_location=device)
    model.load_state_dict(trained_state_dict)
    
    def dataset_accuracy(model: CUB_ResNet_18, data_loader: DataLoader, data_type: str):
        """
        Compute the accuracy based on the given model and dataset.
        
        Args:
        - model: The ResNet-18 model on the CUB dataset.
        - data_loader: The train/test dataloader.
        - data_type: The type of the computation of accuracy, should be in ['train', 'test'].
        """
        model.eval()
        class_correct_top1 = {}
        class_correct_top5 = {}
        class_samples = {}

        total_correct_top1 = 0
        total_correct_top5 = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in tqdm(data_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                _, pred_top1 = outputs.topk(1, 1, True, True)
                _, pred_top5 = outputs.topk(5, 1, True, True)
                pred_top1 = pred_top1.t()
                pred_top5 = pred_top5.t()
                
                correct_top1 = pred_top1.eq(labels.view(1, -1).expand_as(pred_top1))
                correct_top5 = pred_top5.eq(labels.view(1, -1).expand_as(pred_top5))
                
                total_correct_top1 += correct_top1.sum().item()
                total_correct_top5 += correct_top5.sum().item()
                total_samples += labels.size(0)
                # iterate each label
                for label in labels:
                    label = label.item()
                    if label not in class_correct_top1:
                        class_correct_top1[label] = 0
                        class_correct_top5[label] = 0
                        class_samples[label] = 0
                    class_correct_top1[label] += correct_top1[0, labels == label].sum().item()
                    class_correct_top5[label] += correct_top5[:, labels == label].sum().item()
                    class_samples[label] += (labels == label).sum().item()
        
        for label in sorted(class_samples.keys()):
            accuracy_top1 = class_correct_top1[label] / class_samples[label]
            accuracy_top5 = class_correct_top5[label] / class_samples[label]
            print("For the class {:^30} on the CUB dataset, Top-1 accuracy is {:>8.6f}, Top-5 accuracy is {:>8.6f}".format(
                bird_species[label], accuracy_top1, accuracy_top5
            ))

        total_accuracy_top1 = total_correct_top1 / total_samples
        total_accuracy_top5 = total_correct_top5 / total_samples

        print("=" * 120)
        print("For the best model on the CUB dataset, Total Top-1 accuracy is {:>8.6f}, Total Top-5 accuracy is {:>8.6f}".format(
            total_accuracy_top1, total_accuracy_top5
        ))

    # dataset_accuracy(model, train_loader, 'train')
    dataset_accuracy(model, test_loader, 'test')
