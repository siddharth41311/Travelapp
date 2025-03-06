import pandas as pd
import mysql.connector

db = mysql.connector.connect(
    host="localhost",       
    user="root",            
    password="Marijuana@1!",    
    database="travel"   
)

cursor = db.cursor()

#passenger_df = pd.read_excel('PassengerFINALdataset.xlsx', engine='openpyxl')
#flight_df = pd.read_excel('FlightFINALdataset.xlsx', engine='openpyxl')  # Replace with your flight dataset file path
#hotel_df = pd.read_excel('HotelFINALdataset.xlsx', engine='openpyxl') 
#guest_profile_df = pd.read_excel('GuestFINALdataset.xlsx', engine='openpyxl') 
#car_rent_df = pd.read_csv('hotel_reviews', engine='openpyxl')  # Replace with your file path
hotel_reviews = pd.read_csv('Reviews//Hotel_Reviews.csv')  # Replace with your file path


#car_rent_df['Check-in'] = pd.to_datetime(car_rent_df['Check-in'], format='%m/%d/%Y', errors='coerce').dt.strftime('%Y-%m-%d')



#car_rent_df['Check-in'] = pd.to_datetime(car_rent_df['Check-in'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')

# flight_df['Departure_date'] = pd.to_datetime(flight_df['Departure_date'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')

#hotel_df['Check-in'] = pd.to_datetime(hotel_df['Check-in'], errors='coerce').dt.strftime('%Y-%m-%d')

# cursor.execute("SELECT user_id FROM user")
# existing_users = {row[0] for row in cursor.fetchall()} 

# for i, row in passenger_df.iterrows():
#     cursor.execute("""
#         INSERT INTO passenger ( usercode, company, name, gender)
#         VALUES ( %s, %s, %s, %s)
#     """, (row['User_ID'], row['company'], row['Name'], row['gender_x'] ))

# Iterate through the rows of the DataFrame and insert data into MySQL
# for i, row in flight_df.iterrows():
#     cursor.execute("""
#         INSERT INTO flight (travelcode, user_id, departure, arrival, flight_type, flight_price, 
#                             flight_duration, flight_distance, flight_agency, departure_date)
#         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
#     """, (row['travelCode'], row['User_ID'], row['Departure'], row['Arrival'], row['flightType'], 
#           row['Flight_price'], row['Flight_duration'], row['Flight_Distance'], row['Flight_agency'], row['Departure_date']))

# Insert data into 'hotel' table
# for i, row in hotel_df.iterrows():
#     cursor.execute("""
#         INSERT INTO hotel (user_id, travel_code, hotel_name, arrival_place, hotel_stay, hotel_per_day_rent, 
#                            check_in, hotel_total_price)
#         VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
#     """, (row['User_ID'], row['travelCode'], row['Hotel_Name'], row['Arrival_place'], row['Hotel_stay'], 
#           row['Hotel_per_day_price'], row['Check-in'], row['Hotel_TotalPrice']))

# Insert data into 'guest_profile' table
# for i, row in guest_profile_df.iterrows():
#     cursor.execute("""
#         INSERT INTO guest_profile (Guest_Id, TravelCode, Guest_Name, Guest_Gender, Age, 
#                                    Guest_PhoneNo, Guest_Email, IdProof)
#         VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
#     """, (row['Guest_ID'], row['travelCode'], row['Guest_name'], row['Guest_Gender'], row['Age'], 
#           row['Guest_PhoneNo'], row['Guest_email'], row['idProof']))
    
# for i, row in car_rent_df.iterrows():
#     cursor.execute("""
#         INSERT INTO car_rent (User_ID, TravelCode, Rent_Date, Pickup_Location, Dropoff_Location, Car_Type, 
#                               Rental_Agency, Rental_Duration, Car_Total_Distance, Fuel_Policy, Car_BookingStatus, 
#                               Total_Rent_Price)
#         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
#     """, (row['User_ID'], row['travelCode'], row['Check-in'], row['pickupLocation'], row['dropoffLocation'], 
#           row['carType'], row['rentalAgency'], row['rentalDuration'], row['Car_total_distance'], 
#           row['fuelPolicy'], row['Car_bookingStatus'], row['total_rent_price']))

#car_rent_df.drop('Rent_Date',axis=1)

for i, row in hotel_reviews.iterrows():
    cursor.execute("""
        INSERT INTO hotel_review (review)
        VALUES ( %s)
    """, (row['Reviews'],))


db.commit()
cursor.close()
db.close()

print("Data imported successfully.")