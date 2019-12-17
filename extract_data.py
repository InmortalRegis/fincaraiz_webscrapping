from bs4 import BeautifulSoup
import time
import numpy as np
from selenium.webdriver.common.by import By
from selenium import webdriver
options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--incognito')
options.add_argument('--headless')
driver = webdriver.Chrome(
    "/usr/lib/chromium-browser/chromedriver", chrome_options=options)
driver.get("https://www.fincaraiz.com.co/apartamentos/arriendo/bucaramanga/")

def extract_data(page_source):
    areas = []
    precios = []
    habitaciones = []
    soup = BeautifulSoup(page_source, 'lxml')
    apartamentos_selector = soup.find_all('ul', class_='advert')
    for apartamento_selector in apartamentos_selector:
        area = apartamento_selector.find(
            'li', class_='surface li_advert').get_text().split()[0]

        areas.append(float(area.replace(',', '.')))
        habitacion = apartamento_selector.find(
            'li', class_='surface li_advert').get_text().split()[2]

        habitaciones.append(int(habitacion))
        precio = apartamento_selector.find(
            'li', class_='price li_advert').get_text().split()[1].replace('.', '')

        precios.append(float(precio)/1e5)
        # print('Area: ' + str(float(area.replace(',', '.'))) + ' Habt: ' +
        # str(int(habitaciones)) + ' Precio: '+str(float(precio)/1e6))

    pass
    return areas, precios, habitaciones

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


areas = ['areas']
precios = ['precios']
habitaciones = []
items = list(range(0,200))
items_len = len(items)
printProgressBar(0, items_len, prefix = 'Progress:', suffix = 'Complete', length = 50)
for i,item in enumerate(items):
    page_source = driver.page_source
    areas_obtained, precios_obtained, habitaciones_obtained = extract_data(page_source)
    areas += areas_obtained
    precios += precios_obtained
    habitaciones += habitaciones_obtained
    next_button = driver.find_element(
        By.XPATH, '/html/body/div[2]/div[3]/form/div[4]/div[2]/div[1]/div[2]/div/div[3]/div[2]/div/a[7]')
    next_button.click()
    time.sleep(1)
    printProgressBar(i + 1, items_len, prefix = 'Progress:', suffix = 'Complete', length = 50)


# plt.plot(areas[1:], precios[1:], 'o', color="green")
# plt.ylabel('Precio')
# plt.xlabel('Area')
# plt.show()

np.savetxt('aptos_train.csv', [p for p in zip(areas[0:int(len(areas)*0.8)], precios[0:int(len(precios)*0.8)])], delimiter=',', fmt='%s')
np.savetxt('aptos_eval.csv', [p for p in zip(areas[int(len(areas)*0.8):], precios[int(len(precios)*0.8):])], delimiter=',', fmt='%s')