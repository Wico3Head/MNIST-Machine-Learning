import pickle, pygame, sys, threading, random, gzip
import numpy as np
from network import Network
from canvas import Canvas
from setting import *
pygame.init()

train_progress = 0
train_set_size = 70000
retrain_data = 3

with open('network.pkl', 'rb') as f:
    net = pickle.load(f)

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Handwritten Digit Recogniser')
output_label = FONT.render("Output: ", False, 'white')
output_rect = output_label.get_rect(topleft=(770, 60))

clear_board_border = pygame.Rect(700, 370, 250, 50)
clear_board_text = FONT.render("clear board", False, 'white')
clear_board_rect = clear_board_text.get_rect(center=(825, 398))

retrain_ai_border = pygame.Rect(700, 445, 250, 50)
retrain_ai_text = FONT.render("retrain A.I.", False, 'white')
retrain_ai_rect = retrain_ai_text.get_rect(center=(825, 473))

add_entry_border = pygame.Rect(700, 520, 250, 50)
add_entry_text = SMALL_FONT.render("Add to database", False, 'white')
add_entry_rect = add_entry_text.get_rect(center=(825, 548))

remove_entry_border = pygame.Rect(700, 595, 250, 50)
remove_entry_text = SMALL_FONT.render("remove last entry", False, 'white')
remove_entry_rect = remove_entry_text.get_rect(center=(825, 623))

retraining_bg_rect = pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
please_wait_label = FONT.render('Please Wait', False, 'white')
please_wait_rect = please_wait_label.get_rect(center=(500, 280))
please_wait_msg1 = SMALL_FONT.render('The A.I. is training', False, 'white')
please_wait_msg1_rect = please_wait_msg1.get_rect(center=(500, 320))
please_wait_msg2 = SMALL_FONT.render('Testing A.I. accuracy', False, 'white')
please_wait_msg2_rect = please_wait_msg2.get_rect(center=(500, 320))
test_finished_label = FONT.render('Testing Finished', False, 'white')
test_finished_rect = test_finished_label.get_rect(center=(500, 280))
progress_bar_rect = pygame.Rect(350, 350, 300, 40)
program_closing_label = FONT.render('Prorgam is closing...', False, 'white')
program_closing_rect = program_closing_label.get_rect(center=(500, 340))
preparing_label = FONT.render('Preparing Data', False, 'white')
preparing_data_rect = preparing_label.get_rect(center=(500, 340))

add_to_database_menu = pygame.Surface((500, 350))
add_to_database_menu.fill(ADD_TO_DATABASE_MENU_COLOR)

add_to_database_label1 = SMALL_FONT.render('Which number does this ', False, 'white')
add_to_database_label1_rect = add_to_database_label1.get_rect(center=(250, 30))
add_to_database_label2 = SMALL_FONT.render('picture represent?', False, 'white')
add_to_database_label2_rect = add_to_database_label2.get_rect(center=(250, 60))
add_to_database_label3 = SMALL_FONT.render('NOTE: retrain the A.I. after adding data', False, 'white')
add_to_database_label3_rect = add_to_database_label3.get_rect(center=(250, 280))
add_to_database_label4 = SMALL_FONT.render(' to see the effects', False, 'white')
add_to_database_label4_rect = add_to_database_label4.get_rect(center=(250, 310))
x_button = FONT.render('x', False, 'white')
x_rect = x_button.get_rect(center=(470, 30))
zero_button = FONT.render('0', False, 'white')
zero_rect = zero_button.get_rect(center = (90, 140))
one_button = FONT.render('1', False, 'white')
one_rect = one_button.get_rect(center = (170, 140))
two_button = FONT.render('2', False, 'white')
two_rect = two_button.get_rect(center = (250, 140))
three_button = FONT.render('3', False, 'white')
three_rect = three_button.get_rect(center = (330, 140))
four_button = FONT.render('4', False, 'white')
four_rect = four_button.get_rect(center = (410, 140))
five_button = FONT.render('5', False, 'white')
five_rect = five_button.get_rect(center = (90, 220))
six_button = FONT.render('6', False, 'white')
six_rect = six_button.get_rect(center = (170, 220))
seven_button = FONT.render('7', False, 'white')
seven_rect = seven_button.get_rect(center = (250, 220))
eight_button = FONT.render('8', False, 'white')
eight_rect = eight_button.get_rect(center = (330, 220))
nine_button = FONT.render('9', False, 'white')
nine_rect = nine_button.get_rect(center = (410, 220))

add_to_database_menu.blit(add_to_database_label1, add_to_database_label1_rect)
add_to_database_menu.blit(add_to_database_label2, add_to_database_label2_rect)
add_to_database_menu.blit(add_to_database_label3, add_to_database_label3_rect)
add_to_database_menu.blit(add_to_database_label4, add_to_database_label4_rect)
add_to_database_menu.blit(x_button, x_rect)
add_to_database_menu.blit(zero_button, zero_rect)
add_to_database_menu.blit(one_button, one_rect)
add_to_database_menu.blit(two_button, two_rect)
add_to_database_menu.blit(three_button, three_rect)
add_to_database_menu.blit(four_button, four_rect)
add_to_database_menu.blit(five_button, five_rect)
add_to_database_menu.blit(six_button, six_rect)
add_to_database_menu.blit(seven_button, seven_rect)
add_to_database_menu.blit(eight_button, eight_rect)
add_to_database_menu.blit(nine_button, nine_rect)

surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT)) 
surface.set_alpha(180)        
surface.fill('black') 

def retrainAI():
    global net, retraining, train_progress, train_set_size, retrain_data, retrain_state, accuracy, show_time
    with gzip.open('dataset.pkl.gz', 'rb') as f:
        dataset = pickle.load(f, encoding='latin1')
    train_progress = 0
    net = Network([784, 100, 10])
    train_set_size = len(dataset)
    epoch_data = []
    for i in range(retrain_data):
        random.shuffle(dataset)
        for idx in range(train_set_size):
            train_progress += 1
            img = dataset[idx][0]
            expected_output = np.array([0 if dataset[idx][1] != i else 1 for i in range(10)])
            epoch_data.append([img, expected_output])
            if len(epoch_data) == 30:
                net.learn(epoch_data, LEARN_RATE, REGULARISATION_CONSTANT, train_set_size)
                epoch_data.clear()
    if epoch_data:
        net.learn(epoch_data, LEARN_RATE, REGULARISATION_CONSTANT, train_set_size)

    retrain_state = 'test'
    correct = 0
    train_progress = 0
    random.shuffle(dataset)
    for idx in range(train_set_size):
        train_progress += 1
        img = dataset[idx][0]
        outputs = net.activate(img)
        if outputs.tolist().index(max(outputs)) == dataset[idx][1]:
            correct += 1

    retrain_state = 'show'
    accuracy = round(correct / train_set_size * 100, 3)
    show_time = pygame.time.get_ticks()

    with open('network.pkl', 'wb') as f:
        pickle.dump(net, f)
    with open('network.pkl', 'rb') as f:
        net = pickle.load(f) 
    train_progress = 0

def main():
    global retraining, net, retrain_data, train_progress, train_set_size, retrain_state
    canvas = Canvas(screen)
    drawing = False
    retraining = False
    retrain_state = None
    adding = False
    add_label = None
    prev_img = np.full((28, 28), 255, dtype=np.uint8)
    new_data = []

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                screen.blit(surface, (0, 0))
                screen.blit(please_wait_label, please_wait_rect)
                screen.blit(program_closing_label, program_closing_rect)
                pygame.display.update()
                if new_data:
                    with gzip.open('dataset.pkl.gz', 'rb') as f:
                        dataset = pickle.load(f, encoding='latin1')
                    for data in new_data:
                        dataset.append(data)
                    with gzip.open('dataset.pkl.gz', 'wb') as f:
                        pickle.dump(dataset, f)
                    new_data.clear()
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse = event.pos
                if not retraining and not adding:
                    if canvas.onCanvas(mouse):
                        drawing = True
                    elif clear_board_border.collidepoint(mouse):
                        canvas = Canvas(screen)
                    if retrain_ai_border.collidepoint(mouse):
                        if new_data:
                            screen.blit(surface, (0, 0))
                            screen.blit(please_wait_label, please_wait_rect)
                            screen.blit(preparing_label, preparing_data_rect)
                            pygame.display.update()
                            with gzip.open('dataset.pkl.gz', 'rb') as f:
                                dataset = pickle.load(f, encoding='latin1')
                            for data in new_data:
                                dataset.append(data)
                            with gzip.open('dataset.pkl.gz', 'wb') as f:
                                pickle.dump(dataset, f)
                            new_data.clear()
                        retraining = True
                        retrain_ai_thread = threading.Thread(target=retrainAI)
                        retrain_ai_thread.start()
                        retrain_state = 'train'
                    elif add_entry_border.collidepoint(mouse):
                        adding = True
                    elif remove_entry_border.collidepoint(mouse):
                        if not new_data:
                            screen.blit(surface, (0, 0))
                            screen.blit(please_wait_label, please_wait_rect)
                            pygame.display.update()
                            with gzip.open('dataset.pkl.gz', 'rb') as f:
                                dataset = pickle.load(f, encoding='latin1')
                            dataset = dataset[:-1]
                            with gzip.open('dataset.pkl.gz', 'wb') as f:
                                pickle.dump(dataset, f)
                        else:
                            new_data = new_data[:-1]

                elif adding:
                    mouse = (mouse[0] - 250, mouse[1] - 175)
                    if x_rect.collidepoint(mouse):
                        adding = False
                    elif zero_rect.collidepoint(mouse):
                        add_label = 0
                    elif one_rect.collidepoint(mouse):
                        add_label = 1
                    elif two_rect.collidepoint(mouse):
                        add_label = 2
                    elif three_rect.collidepoint(mouse):
                        add_label = 3
                    elif four_rect.collidepoint(mouse):
                        add_label = 4
                    elif five_rect.collidepoint(mouse):
                        add_label = 5
                    elif six_rect.collidepoint(mouse):
                        add_label = 6
                    elif seven_rect.collidepoint(mouse):
                        add_label = 7
                    elif eight_rect.collidepoint(mouse):
                        add_label = 8
                    elif nine_rect.collidepoint(mouse):
                        add_label = 9

            if event.type == pygame.MOUSEBUTTONUP:
                if drawing:
                    drawing = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE and not retraining and not adding:
                    canvas = Canvas(screen)


        if not retraining and not adding:
            if drawing:
                mouse = pygame.mouse.get_pos()
                if not canvas.onCanvas(mouse):
                    drawing = False
                else:
                    canvas.paint(mouse)

            if False in (canvas.low_res_image == prev_img):
                outputs = net.activate(canvas.low_res_image.flatten() / 255)
                outputs = sorted([(i, outputs[i]/sum(outputs)) for i in range(10)], key=lambda x:x[1], reverse=True)
                prev_img = canvas.low_res_image

        screen.fill(BG_COLOR)
        canvas.draw()
    
        screen.blit(output_label, output_rect)
        for index, (number, confidence) in enumerate(outputs):
            label = FONT.render(f'{number}: {round(confidence*100, 2)}%', False, 'white')
            label_rect = label.get_rect(topleft=(690 + 160 * (index // 5), 110 + 50 * (index % 5)))
            screen.blit(label, label_rect)

        pygame.draw.rect(screen, 'white', clear_board_border, 3)
        screen.blit(clear_board_text, clear_board_rect)
        pygame.draw.rect(screen, 'white', retrain_ai_border, 3)
        screen.blit(retrain_ai_text, retrain_ai_rect)
        pygame.draw.rect(screen, 'white', add_entry_border, 3)
        screen.blit(add_entry_text, add_entry_rect)   
        pygame.draw.rect(screen, 'white', remove_entry_border, 3)
        screen.blit(remove_entry_text, remove_entry_rect)
        
        if retraining:    
            screen.blit(surface, (0,0))    

            if retrain_state == 'train':
                screen.blit(please_wait_label, please_wait_rect)
                screen.blit(please_wait_msg1, please_wait_msg1_rect)
                pygame.draw.rect(screen, 'white', progress_bar_rect, 3)
                progress_bar = pygame.Rect(353, 353, train_progress / (train_set_size * retrain_data) * 294, 34)     
                pygame.draw.rect(screen, 'white', progress_bar)   
            elif retrain_state == 'test':
                screen.blit(please_wait_label, please_wait_rect)
                screen.blit(please_wait_msg2, please_wait_msg2_rect)
                pygame.draw.rect(screen, 'white', progress_bar_rect, 3)
                progress_bar = pygame.Rect(353, 353, train_progress / train_set_size * 294, 34)     
                pygame.draw.rect(screen, 'white', progress_bar)  
            else: 
                if pygame.time.get_ticks() - show_time >= 1500:
                    retraining = False
                accuracy_label = FONT.render(f'Test Accuracy: {accuracy}%', False, 'white')
                accuracy_rect = accuracy_label.get_rect(center=(500, 320))
                screen.blit(test_finished_label, test_finished_rect)
                screen.blit(accuracy_label, accuracy_rect)

        elif adding:
            screen.blit(surface, (0,0))  
            screen.blit(add_to_database_menu, (250, 175))
            if add_label != None:
                new_data.append((np.array(canvas.low_res_image.flatten() / 255, dtype=np.float32), add_label))
                add_label = None
                adding = False
                canvas = Canvas(screen)

        pygame.display.update()

if __name__ == "__main__":
    main()