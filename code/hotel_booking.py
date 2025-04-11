import pandas as pd
import mysql.connector

db = mysql.connector.connect(
    host="localhost",       
    user="root",            
    password="Marijuana@1!",    
    database="TravelDB"   
)

cursor = db.cursor()

# Hotel_booking = pd.read_csv('datas//New_Hotel_data.csv')

# for i, row in Hotel_booking.iterrows():
#     cursor.execute("""
#         INSERT INTO HotelBookings (travelCode, User_ID, Departure, Arrival, Assigned_Hotel, Stars, Check_in_Date, 
#                                    Bedroom_Type, Room_Price_per_Night, Number_of_Adults, Number_of_Children, 
#                                    Number_of_Bedrooms, Total_Cost_per_Night, Amenities, Days_of_Stay, 
#                                    Total_Cost, Check_Out_Date)
#         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
#     """, (row['travelCode'], row['User_ID'], row['Departure'], row['Arrival'], row['Assigned Hotel'], row['Stars'], 
#           row['Check-in Date'], row['Bedroom Type'], row['Room Price per Night'], row['Number of Adults'], 
#           row['Number of Children'], row['Number of Bedrooms'], row['Total Cost per Night'], row['Amenities'], 
#           row['Days of Stay'], row['Total Cost'], row['Check-Out Date']))

Hotel_booking = pd.read_excel('datas//Final_FRHotel_data.xlsx',engine='openpyxl')

# for i, row in Hotel_booking.iterrows():
#     cursor.execute("""
#         INSERT INTO HotelBooking (travelCode, User_ID, Departure, Arrival, Hotel, Stars, Check_in_Date, 
#                                    Bedroom_Type, Room_Price_per_Night, Number_of_Adults, Number_of_Children, 
#                                    Number_of_Bedrooms, Total_Price_per_Night, Amenities, Days_of_Stay, 
#                                    Total_Cost, Check_Out_Date)
#         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
#     """, (row['travelCode'], row['User_ID'], row['Departure'], row['Arrival'], row['Hotel'], row['Stars'], 
#           row['Check-in'], row['Bedroom Type'], row['Room Price per Night'], row['Number of Adults'], 
#           row['Number of Children'], row['Number of Bedrooms'], row['Total Price per Night'], row['Amenities'], 
#           row['Days of Stay'], row['Total Cost'], row['Check-Out']))

for i, row in Hotel_booking.iterrows():
    cursor.execute("""
        INSERT INTO HotelBookingss (travelCode, User_ID, Departure, Arrival, Hotel, Stars, Check_in_Date, 
                                   Bedroom_Type, Number_of_Adults, Number_of_Children, 
                                   Number_of_Bedrooms, Days_of_Stay, Check_Out_Date)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (row['travelCode'], row['User_ID'], row['Departure'], row['Arrival'], row['Hotel'], row['Stars'], 
          row['Check-in'], row['Bedroom Type'], row['Number of Adults'], 
          row['Number of Children'], row['Number of Bedrooms'], 
          row['Days of Stay'], row['Check-Out']))

db.commit()
cursor.close()
db.close()

print("Data imported successfully.")