Show databases;
use travel;
CREATE TABLE passenger (
    usercode INT PRIMARY KEY ,  
    company VARCHAR(255),                    
    name VARCHAR(255),                       
    gender ENUM('Male', 'Female', 'Other')   
);
desc passenger;
ALTER TABLE passenger MODIFY gender VARCHAR(10);
SELECT * FROM passenger;

CREATE TABLE flight (
    travelcode INT PRIMARY KEY, 
    user_id INT,
    departure VARCHAR(100),
    arrival VARCHAR(100),
    flight_type VARCHAR(50),
    flight_price DECIMAL(10, 2),
    flight_duration INT, 
    flight_distance INT, 
    flight_agency VARCHAR(100),
    departure_date DATETIME,
    FOREIGN KEY (user_id) REFERENCES passenger(usercode)
);
SELECT * FROM flight;

CREATE TABLE hotel (
    user_id INT,
    travel_code INT ,
    hotel_name VARCHAR(100),
    arrival_place VARCHAR(100),
    hotel_stay INT, 
    hotel_per_day_rent DECIMAL(10, 2),
    check_in DATETIME,
    hotel_total_price DECIMAL(10, 2),
    FOREIGN KEY (travel_code) REFERENCES flight(travelcode)
);

SELECT * FROM hotel;

CREATE TABLE guest_profile (
    Guest_Id INT,
    TravelCode INT,
    Guest_Name VARCHAR(100),
    Guest_Gender VARCHAR(10),
    Age INT,
    Guest_PhoneNo VARCHAR(50),
    Guest_Email VARCHAR(100),
    IdProof VARCHAR(50),
    FOREIGN KEY (TravelCode) REFERENCES flight(TravelCode)
);

CREATE TABLE car_rent (
    Rent_ID INT AUTO_INCREMENT PRIMARY KEY,
    User_ID INT,  
    TravelCode INT,  
    Rent_Date DATETIME,
    Pickup_Location VARCHAR(255) NOT NULL,
    Dropoff_Location VARCHAR(255) NOT NULL,
    Car_Type VARCHAR(100) NOT NULL,
    Rental_Agency VARCHAR(100) NOT NULL,
    Rental_Duration INT NOT NULL,  
    Car_Total_Distance DECIMAL(10,2) NOT NULL,  
    Fuel_Policy VARCHAR(50) NOT NULL,  
    Car_BookingStatus VARCHAR(50) NOT NULL, 
    Total_Rent_Price DECIMAL(10,2) NOT NULL,  


    FOREIGN KEY (User_ID) REFERENCES passenger(usercode),
    FOREIGN KEY (TravelCode) REFERENCES flight(travelcode)
);

CREATE TABLE hotel_review (
    review TEXT NOT NULL
);
SELECT * FROM hotel_review