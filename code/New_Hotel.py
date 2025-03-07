import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

hotels = {
    'Jaipur': [
        'Hotel Kalyan',                         # ⭐ 1-star
        'Hotel Shikha',                         # ⭐⭐ 2-star
        'Vesta International',                  # ⭐⭐⭐ 3-star
        'Holiday Inn Jaipur City Centre',       # ⭐⭐⭐⭐ 4-star
        'The Oberoi Rajvilas'                   # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Udaipur': [
        'Hotel Dream Palace',                   # ⭐ 1-star
        'Hotel Minerwa',                        # ⭐⭐ 2-star
        'Hotel Lakend',                         # ⭐⭐⭐ 3-star
        'The Ananta Udaipur',                   # ⭐⭐⭐⭐ 4-star
        'Taj Lake Palace'                       # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Kota': [
        'Hotel Madhushree',                     # ⭐ 1-star
        'Hotel Surya Prime',                    # ⭐⭐ 2-star
        'Hotel Lilac',                          # ⭐⭐⭐ 3-star
        'Hotel The Grand Chandiram',            # ⭐⭐⭐⭐ 4-star
        'Country Inn & Suites'                  # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Gangtok': [
        'Hotel Karma',                          # ⭐ 1-star
        'Hotel Tashi Delek',                    # ⭐⭐ 2-star
        'The Royal Plaza',                      # ⭐⭐⭐ 3-star
        'WelcomHeritage Denzong Regency',       # ⭐⭐⭐⭐ 4-star
        'Mayfair Spa Resort & Casino'           # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Coimbatore': [
        'Hotel City Tower',                     # ⭐ 1-star
        'Hotel The Arcadia',                    # ⭐⭐ 2-star
        'Gokulam Park Coimbatore',              # ⭐⭐⭐ 3-star
        'Le Meridien Coimbatore',               # ⭐⭐⭐⭐ 4-star
        'Vivanta Coimbatore'                    # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Chennai': [
        'Hotel Pandian',                        # ⭐ 1-star
        'Hotel Marina Inn',                     # ⭐⭐ 2-star
        'Green Park Hotel',                     # ⭐⭐⭐ 3-star
        'The Accord Metropolitan',              # ⭐⭐⭐⭐ 4-star
        'ITC Grand Chola'                       # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Hyderabad': [
        'Hotel Grand Elite',                    # ⭐ 1-star
        'Hotel Geetanjali',                     # ⭐⭐ 2-star
        'Lemon Tree Hotel',                     # ⭐⭐⭐ 3-star
        'Radisson Blu Plaza',                   # ⭐⭐⭐⭐ 4-star
        'Taj Falaknuma Palace'                  # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Khowai': [
        'Hotel Green View',                     # ⭐ 1-star
        'Hotel Gitanjali',                      # ⭐⭐ 2-star
        'Hotel Royal Residency',                # ⭐⭐⭐ 3-star
        'Hotel White House',                    # ⭐⭐⭐⭐ 4-star
        'Hotel Green Park'                      # ⭐⭐⭐⭐ 4-star (Alternative for 5-star)
    ],
    'Ayodhya': [
        'Hotel Panchsheel',                     # ⭐ 1-star
        'Hotel Krishna Palace',                 # ⭐⭐ 2-star
        'Taraji Resort',                        # ⭐⭐⭐ 3-star
        'Shree Ram Hotel',                      # ⭐⭐⭐⭐ 4-star
        'Hotel Ramprastha'                      # ⭐⭐⭐⭐ 4-star (Alternative for 5-star)
    ],
    'Varanasi': [
        'Hotel Buddha',                         # ⭐ 1-star
        'Hotel Tridev',                         # ⭐⭐ 2-star
        'Hotel Hindusthan International',       # ⭐⭐⭐ 3-star
        'Ramada Plaza JHV',                     # ⭐⭐⭐⭐ 4-star
        'BrijRama Palace'                       # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Prayagraj': [
        'Hotel Yatrik',                         # ⭐ 1-star
        'Hotel Milan Palace',                   # ⭐⭐ 2-star
        'Hotel Harsh Ananda',                   # ⭐⭐⭐ 3-star
        'Grand Continental Hotel',              # ⭐⭐⭐⭐ 4-star
        'Kanha Shyam Hotel'                     # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Dehradun': [
        'Hotel Abhinandan',                     # ⭐ 1-star
        'Hotel Softel Plaza',                   # ⭐⭐ 2-star
        'Four Points by Sheraton',              # ⭐⭐⭐ 3-star
        'Lemon Tree Hotel',                     # ⭐⭐⭐⭐ 4-star
        'JW Marriott Mussoorie'                 # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Kolkata': [
        'Hotel Emirates',                       # ⭐ 1-star
        'Hotel Dee Empresa',                    # ⭐⭐ 2-star
        'The Peerless Inn',                     # ⭐⭐⭐ 3-star
        'The Park Kolkata',                     # ⭐⭐⭐⭐ 4-star
        'The Oberoi Grand'                      # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Delhi NCR': [
        'Hotel Le Roi',                         # ⭐ 1-star
        'Hotel Godwin Deluxe',                  # ⭐⭐ 2-star
        'The Hans',                             # ⭐⭐⭐ 3-star
        'The Lalit New Delhi',                  # ⭐⭐⭐⭐ 4-star
        'The Leela Palace'                      # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Chandigarh': [
        'Hotel Park Inn',                       # ⭐ 1-star
        'Hotel Oyster',                         # ⭐⭐ 2-star
        'Hometel Chandigarh',                   # ⭐⭐⭐ 3-star
        'JW Marriott Chandigarh',               # ⭐⭐⭐⭐ 4-star
        'Taj Chandigarh'                        # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Puducherry': [
        'Hotel Coramandal Heritage',            # ⭐ 1-star
        'Hotel Annamalai International',        # ⭐⭐ 2-star
        'Le Dupleix',                           # ⭐⭐⭐ 3-star
        'The Promenade',                        # ⭐⭐⭐⭐ 4-star
        'The Residency Towers'                  # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Diu': [
        'Hotel Palacio De Diu',                 # ⭐ 1-star
        'Hotel Apaar',                          # ⭐⭐ 2-star
        'Kostamar Beach Resort',                # ⭐⭐⭐ 3-star
        'Hotel Kohinoor',                       # ⭐⭐⭐⭐ 4-star
        'Radhika Beach Resort'                  # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Srinagar': [
        'Hotel New Green View',                 # ⭐ 1-star
        'Hotel Ahdoos',                         # ⭐⭐ 2-star
        'Hotel Comrade Inn',                    # ⭐⭐⭐ 3-star
        'Vivanta Dal View',                     # ⭐⭐⭐⭐ 4-star
        'The Lalit Grand Palace'                # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Jamshedpur': [
        'Hotel South Park',              # ⭐ 1-star
        'The Boulevard Hotel',           # ⭐⭐ 2-star
        'Hotel Ganga Regency',           # ⭐⭐⭐ 3-star
        'Ramada Jamshedpur',             # ⭐⭐⭐⭐ 4-star
        'Alcor Hotel'                    # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Ranchi': [
        'Hotel Anjali',                  # ⭐ 1-star
        'Hotel Accord',                  # ⭐⭐ 2-star
        'Capitol Residency',             # ⭐⭐⭐ 3-star
        'Radisson Blu Hotel Ranchi',     # ⭐⭐⭐⭐ 4-star
        'Chanakya BNR Hotel'             # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Bengaluru': [
        'Hotel Empire',                  # ⭐ 1-star
        'Hotel Nandhini',                # ⭐⭐ 2-star
        'Treebo Trend Raj Premier',      # ⭐⭐⭐ 3-star
        'The Chancery Pavilion',         # ⭐⭐⭐⭐ 4-star
        'The Leela Palace Bengaluru'     # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Mysuru': [
        'Hotel Roopa',                   # ⭐ 1-star
        'Hotel Aditya',                  # ⭐⭐ 2-star
        'The Quorum',                    # ⭐⭐⭐ 3-star
        'Country Inn & Suites',          # ⭐⭐⭐⭐ 4-star
        'Radisson Blu Plaza Hotel Mysore' # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Thiruvananthapuram': [
        'Hotel Blue Nest',               # ⭐ 1-star
        'Hotel Prathiba Heritage',       # ⭐⭐ 2-star
        'Keys Select Hotel',             # ⭐⭐⭐ 3-star
        'Mascot Hotel',                  # ⭐⭐⭐⭐ 4-star
        'Hilton Garden Inn'              # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Gwalior': [
        'Hotel Grace',                   # ⭐ 1-star
        'Hotel Radiance',                # ⭐⭐ 2-star
        'The Central Park',              # ⭐⭐⭐ 3-star
        'Clarks Inn Suites',             # ⭐⭐⭐⭐ 4-star
        'Taj Usha Kiran Palace'          # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Indore': [
        'Hotel Paradise',                # ⭐ 1-star
        'Hotel Kalinga',                 # ⭐⭐ 2-star
        'Effotel Hotel',                 # ⭐⭐⭐ 3-star
        'Radisson Blu Hotel Indore',     # ⭐⭐⭐⭐ 4-star
        'Sayaji Hotel'                   # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Bhopal': [
        'Hotel Gaurav Palace',           # ⭐ 1-star
        'Hotel Shree Vatika',            # ⭐⭐ 2-star
        'Courtyard by Marriott',         # ⭐⭐⭐ 3-star
        'Jehan Numa Palace Hotel',       # ⭐⭐⭐⭐ 4-star
        'Taj Lakefront Bhopal'           # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Mumbai': [
        'Hotel New Shree Niwas',         # ⭐ 1-star
        'Hotel City Palace',             # ⭐⭐ 2-star
        'Residency Hotel',               # ⭐⭐⭐ 3-star
        'The Orchid Mumbai',             # ⭐⭐⭐⭐ 4-star
        'The Taj Mahal Palace'           # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Navi Mumbai': [
        'Hotel Yogi Midtown',            # ⭐ 1-star
        'Hotel Abbott',                  # ⭐⭐ 2-star
        'Royal Orchid Central Grazia',   # ⭐⭐⭐ 3-star
        'The Regenza by Tunga',          # ⭐⭐⭐⭐ 4-star
        'Four Points by Sheraton'        # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Nashik': [
        'Hotel Panchavati',              # ⭐ 1-star
        'Hotel Sai Palace',              # ⭐⭐ 2-star
        'Express Inn',                   # ⭐⭐⭐ 3-star
        'The Gateway Hotel',             # ⭐⭐⭐⭐ 4-star
        'Ginger Nashik'                  # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Pune': [
        'Hotel Shivkrupa',               # ⭐ 1-star
        'Hotel Orchard',                 # ⭐⭐ 2-star
        'St Laurn Hotel',                # ⭐⭐⭐ 3-star
        'Hyatt Pune',                    # ⭐⭐⭐⭐ 4-star
        'Conrad Pune'                    # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Imphal': [
        'Hotel Krishtina',               # ⭐ 1-star
        'Hotel Imphal',                  # ⭐⭐ 2-star
        'The Classic Hotel',             # ⭐⭐⭐ 3-star
        'Hotel Yaiphaba',                # ⭐⭐⭐⭐ 4-star
        'Classic Grande Imphal'          # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Shillong': [
        'Hotel Pine Hill',               # ⭐ 1-star
        'Hotel Pegasus Crown',           # ⭐⭐ 2-star
        'Hotel Centre Point',            # ⭐⭐⭐ 3-star
        'Ri Kynjai',                     # ⭐⭐⭐⭐ 4-star
        'Tripura Castle'                 # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Puri': [
        'Hotel Sonar Bangla',            # ⭐ 1-star
        'Hotel Gandhara',                # ⭐⭐ 2-star
        'Hotel Holiday Resort',          # ⭐⭐⭐ 3-star
        'MAYFAIR Waves',                 # ⭐⭐⭐⭐ 4-star
        'Toshali Sands'                  # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Bhubaneswar': [
        'Hotel Arya Palace',             # ⭐ 1-star
        'Hotel Suryansh',                # ⭐⭐ 2-star
        'VITS Bhubaneswar',              # ⭐⭐⭐ 3-star
        'Swosti Premium',                # ⭐⭐⭐⭐ 4-star
        'Mayfair Lagoon'                 # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Ludhiana': [
        'Hotel Nanda',                   # ⭐ 1-star
        'Hotel Maharaja Regency',        # ⭐⭐ 2-star
        'Hotel Gulmor',                  # ⭐⭐⭐ 3-star
        'Radisson Blu Hotel Ludhiana',   # ⭐⭐⭐⭐ 4-star
        'Hyatt Regency Ludhiana'         # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Amritsar': [
        'Hotel City Castle',             # ⭐ 1-star
        'Hotel CJ International',        # ⭐⭐ 2-star
        'Hotel Hong Kong Inn',           # ⭐⭐⭐ 3-star
        'Holiday Inn Amritsar',          # ⭐⭐⭐⭐ 4-star
        'Taj Swarna Amritsar'            # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Tirupati': [
        'Hotel Sandeep Residency',            # ⭐ 1-star
        'Hotel Mayura',                       # ⭐⭐ 2-star
        'Treebo Trend SLS Grand',             # ⭐⭐⭐ 3-star
        'Fortune Select Grand Ridge',         # ⭐⭐⭐⭐ 4-star
        'Marasa Sarovar Premiere'             # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Visakhapatnam': [
        'Hotel Lakshmi Grand',                # ⭐ 1-star
        'Hotel Akshaya',                      # ⭐⭐ 2-star
        'Treebo Trend Seaesta RK Beach',      # ⭐⭐⭐ 3-star
        'The Gateway Hotel Beach Road',       # ⭐⭐⭐⭐ 4-star
        'Novotel Visakhapatnam Varun Beach'   # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Vijayawada': [
        'Hotel Manorama',                    # ⭐ 1-star
        'Innotel Hotel',                     # ⭐⭐ 2-star
        'The Kay Hotel',                     # ⭐⭐⭐ 3-star
        'Quality Hotel DV Manor',            # ⭐⭐⭐⭐ 4-star
        'Novotel Vijayawada Varun'           # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Itanagar': [
        'Hotel Blue Pine',                   # ⭐ 1-star
        'Hotel Todo',                        # ⭐⭐ 2-star
        'Hotel Arun Subansiri',              # ⭐⭐⭐ 3-star
        'Hotel Donyi Polo Ashok',            # ⭐⭐⭐⭐ 4-star
        'Hotel PYBSS'                        # ⭐⭐⭐⭐ 4-star (Alternative for 5-star)
    ],
    'Dhubri': [
        'Hotel Green View',                  # ⭐ 1-star
        'Hotel Highway',                     # ⭐⭐ 2-star
        'Hotel Executive Inn',               # ⭐⭐⭐ 3-star
        'Hotel Geetanjali',                  # ⭐⭐⭐⭐ 4-star
        'Hotel Rajmahal'                     # ⭐⭐⭐⭐ 4-star (Alternative for 5-star)
    ],
    'Guwahati': [
        'Hotel Siroy Lily',                  # ⭐ 1-star
        'Hotel Atithi',                      # ⭐⭐ 2-star
        'Hotel Kiranshree Portico',          # ⭐⭐⭐ 3-star
        'Hotel Dynasty',                     # ⭐⭐⭐⭐ 4-star
        'Radisson Blu Hotel Guwahati'        # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Patna': [
        'Hotel Chanakya Inn',                # ⭐ 1-star
        'Hotel Samrat International',        # ⭐⭐ 2-star
        'Hotel Patliputra Exotica',          # ⭐⭐⭐ 3-star
        'Hotel Maurya Patna',                # ⭐⭐⭐⭐ 4-star
        'Lemon Tree Premier Patna'           # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Raipur': [
        'Hotel Simran',                      # ⭐ 1-star
        'Hotel Grand Arjun',                 # ⭐⭐ 2-star
        'Hotel Babylon Inn',                 # ⭐⭐⭐ 3-star
        'Hotel VW Canyon',                   # ⭐⭐⭐⭐ 4-star
        'Courtyard by Marriott Raipur'       # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Bilaspur': [
        'Hotel Shree Shyam International',   # ⭐ 1-star
        'Hotel East Park',                   # ⭐⭐ 2-star
        'Hotel Intercity International',     # ⭐⭐⭐ 3-star
        'Hotel Heavens Park',                # ⭐⭐⭐⭐ 4-star
        'Hotel Central Point International'  # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Raigarh': [
        'Hotel Shreshtha',                   # ⭐ 1-star
        'Hotel Jindal Regency',              # ⭐⭐ 2-star
        'Hotel Ans International',           # ⭐⭐⭐ 3-star
        'Hotel Trinity Grand',               # ⭐⭐⭐⭐ 4-star
        'Hotel Shubham Palace'               # ⭐⭐⭐⭐ 4-star (Alternative for 5-star)
    ],
    'Mopa': [
        'Hotel Hill Rock',                   # ⭐ 1-star
        'Hotel Swarnam',                     # ⭐⭐ 2-star
        'Hotel La Grace',                    # ⭐⭐⭐ 3-star
        'Hotel Hill Rock Suites',            # ⭐⭐⭐⭐ 4-star
        'Hotel Swarnam Deluxe'               # ⭐⭐⭐⭐ 4-star (Alternative for 5-star)
    ],
    'Dabolim': [
        'Hotel Cliff',                       # ⭐ 1-star
        'Hotel La Paz Gardens',              # ⭐⭐ 2-star
        'The HQ',                            # ⭐⭐⭐ 3-star
        'Bogmallo Beach Resort',             # ⭐⭐⭐⭐ 4-star
        'Coconut Creek Resort'               # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Ahmedabad': [
        'Hotel Kamran Palace',               # ⭐ 1-star
        'Hotel Host Inn',                    # ⭐⭐ 2-star
        'Treebo Trend Ambassador',           # ⭐⭐⭐ 3-star
        'Fortune Landmark',                  # ⭐⭐⭐⭐ 4-star
        'Hyatt Regency Ahmedabad'            # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Surat': [
        'Hotel Merit',                       # ⭐ 1-star
        'Hotel Central Excellency',          # ⭐⭐ 2-star
        'Lords Plaza Surat',                 # ⭐⭐⭐ 3-star
        'The Grand Bhagwati',                # ⭐⭐⭐⭐ 4-star
        'Surat Marriott Hotel'               # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Vadodara': [
        'Hotel Skylight',                    # ⭐ 1-star
        'Hotel Alpha',                       # ⭐⭐ 2-star
        'Hampton by Hilton Vadodara',        # ⭐⭐⭐ 3-star
        'Four Points by Sheraton Vadodara',  # ⭐⭐⭐⭐ 4-star
        'The Fern Residency Vadodara'        # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Hisar': [
        'Hotel City Square',                 # ⭐ 1-star
        'Hotel Relax',                       # ⭐⭐ 2-star
        'Hotel Grace',                       # ⭐⭐⭐ 3-star
        'Hotel Saffron',                     # ⭐⭐⭐⭐ 4-star
        'WelcomHotel by ITC Hisar'           # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Shimla': [
        'Hotel Dreamland',                   # ⭐ 1-star
        'Hotel Shingar',                     # ⭐⭐ 2-star
        'Hotel Marina',                      # ⭐⭐⭐ 3-star
        'Radisson Hotel Shimla',             # ⭐⭐⭐⭐ 4-star
        'Wildflower Hall, An Oberoi Resort'  # ⭐⭐⭐⭐⭐ 5-star
    ],
    'Kullu-Manali': [
        'Hotel Greenfields',                 # ⭐ 1-star
        'Hotel Snow Park',                   # ⭐⭐ 2-star
        'Hotel Piccadilly',                  # ⭐⭐⭐ 3-star
        'Apple Country Resorts',             # ⭐⭐⭐⭐ 4-star
        'The Himalayan'                      # ⭐⭐⭐⭐⭐ 5-star
    ],
}

# Convert to DataFrame
df_hotels = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in hotels.items()]))
print(df_hotels)

df = pd.DataFrame(hotels)
print(df)

hotel_df = pd.read_excel('datas//Final_India.xlsx')
print(hotel_df)