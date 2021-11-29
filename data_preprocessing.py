import pandas
import pandas as pd
import json

if __name__ == "__main__":
    df = pandas.read_json('data/yelp_academic_dataset_business.json', lines=True)
    austin = df[df['city'] == 'Austin']#.to_csv('Austin.csv')
    orlando = df[df['city'] == 'Orlando']#.to_csv('Orlando.csv')
    orlando['categories'] = orlando['categories'].map(
        lambda categories: '' if type(categories) is not str else categories.split(',')[0])
    austin['categories'] = austin['categories'].map(
        lambda categories: '' if type(categories) is not str else categories.split(',')[0])

    categories = json.load(open('categories.json'))
    cats_dict = dict()
    for cat in categories:
        if len(cat['parents']) > 0:
            cats_dict[cat['title']] = cat['parents'][0]
            cats_dict[cat['alias']] = cat['parents'][0]

    cont = True
    while cont:
        cont = False
        for k, v in cats_dict.items():
            if v in cats_dict:
                cats_dict[k] = cats_dict[v]
                cont = True


    orlando['categories'] = orlando['categories'].map(
        lambda categories: '' if type(categories) is not str else categories.split(',')[0])
    austin['categories'] = austin['categories'].map(
        lambda categories: '' if type(categories) is not str else categories.split(',')[0])

    cats_dict['Restaurants'] = 'restaurants'
    cats_dict['Food'] = 'food'
    cats_dict['Shopping'] = 'shopping'
    cats_dict['Home Services'] = 'homeservices'
    cats_dict['Local Services'] = 'shopping'
    cats_dict['Local Services'] = 'localservices'
    cats_dict['Arts & Entertainment'] = 'arts'
    cats_dict['Shopping'] = 'shopping'
    cats_dict['Financial Services'] = 'financialservices'
    cats_dict['Education'] = 'education'
    cats_dict['Ethnic Food'] = 'food'
    cats_dict['Mass Media'] = 'massmedia'
    cats_dict['Arabian'] = 'food'
    cats_dict['Beauty & Spas'] = 'beautysvc'
    cats_dict['Health & Medical'] = 'health'
    cats_dict['Nightlife'] = 'nightlife'
    cats_dict['Beer'] = 'food'
    cats_dict['Public Services & Government'] = 'publicservicesgovt'
    cats_dict['Religious Organizations'] = 'religiousorgs'
    cats_dict['Hotels & Travel'] = 'hotelstravel'
    cats_dict['Active Life'] = 'active'
    cats_dict['Event Planning & Services'] = 'eventservices'
    cats_dict['Pets'] = 'pets'
    cats_dict['Professional Services'] = 'professional'
    cats_dict['localflavor'] = 'food'
    cats_dict['Ethnic Grocery'] = 'shopping'
    cats_dict['Used'] = 'shopping'
    cats_dict['Local Flavor'] = 'food'
    cats_dict['Dry Cleaning & Laundry'] = 'homeservices'
    cats_dict['Automotive'] = 'auto'

    orlando['categories'] = orlando['categories'].map(
        lambda cat: cats_dict[cat] if cat in cats_dict else cat)
    austin['categories'] = austin['categories'].map(
        lambda cat: cats_dict[cat] if cat in cats_dict else cat)


    orlando.to_csv('orlando_cats.csv')
    austin.to_csv('austin_cats.csv')



