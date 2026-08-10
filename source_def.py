import numpy as np
import argparse  # For parsing command-line arguments


# Function to convert Right Ascension (RA) from hours-minutes-seconds (h:m:s) format to radians
def ra_conversion(RA):
    # Split the RA string into hours, minutes, and seconds
    ra = RA.split(':')
    
    # Convert seconds, minutes, and hours into a fractional value in hours
    ra_s = float(ra[-1]) / 3600  # Convert seconds to fractional hours
    ra_m = float(ra[1]) / 60     # Convert minutes to fractional hours
    ra_h = float(ra[0])          # Hours (already in hours)
    
    # Convert total RA in hours to radians (1 hour = pi/12 radians)
    Ra = (ra_s + ra_m + ra_h) * (np.pi / 12)
    return Ra


# Function to convert Declination (DEC) from degrees-minutes-seconds (d:m:s) format to radians
def dec_conversion(DEC):
    # Split the DEC string into degrees, minutes, and seconds
    dec = DEC.split(':')
    
    # Convert seconds, minutes, and degrees into a fractional value in degrees
    dec_s = float(dec[-1]) / 3600  # Convert seconds to fractional degrees
    dec_m = float(dec[1]) / 60     # Convert minutes to fractional degrees
    dec_d = float(dec[0])          # Degrees (already in degrees)
    
    # Convert total DEC in degrees to radians
    Dec = np.deg2rad((dec_d + dec_m + dec_s))
    return Dec


# Function to define the unit vectors u, v, w for the source direction in the sky
def uvw_cap(ra, dec):
    # Define the unit vector pointing towards the North Pole
    n_cap = np.array([[0, 0, 1]])  # North pole direction
    
    # Define the unit vector pointing towards the source direction (based on RA and DEC)
    w_cap = np.array([[(np.cos(dec) * np.cos(ra)),  
                       (np.cos(dec) * np.sin(ra)),    
                       (np.sin(dec))]])             
    
    # Define the u vector (perpendicular to both the w vector and n_cap)
    u_cap = np.cross(n_cap, w_cap)  # Cross product of n_cap and w_cap to get u_cap
    u_cap = u_cap / np.linalg.norm(u_cap)  # Normalize the u vector (make it a unit vector)
    
    # Define the v vector (perpendicular to both the w vector and u vector)
    v_cap = np.cross(w_cap, u_cap)  # Cross product of w_cap and u_cap to get v_cap
    v_cap = v_cap / np.linalg.norm(v_cap)  # Normalize the v vector (make it a unit vector)
    
    # Return the three unit vectors u, v, and w
    return u_cap, v_cap, w_cap


# Uncomment this block if you want to run this script from the command line
# if __name__ == '__main__':
    
#     # Set up argument parsing to take RA and DEC from the command line
#     parse = argparse.ArgumentParser()
#     parse.add_argument('RA', type=str, help='Provide the Right Ascension of the source in "h:m:s" format')
#     parse.add_argument('DEC', type=str, help='Provide the Declination of the source in "d:m:s" format')

#     # Parse the arguments
#     args = parse.parse_args()

#     # Convert RA and DEC to radians
#     ra = ra_conversion(args.RA)
#     dec = dec_conversion(args.DEC)
    
#     # Print the calculated u, v, w unit vectors
#     print(uvw_cap(ra, dec))
