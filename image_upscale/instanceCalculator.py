def quick_instance_calc(images_per_year, minutes_per_image):
    # based on example
    # 1million images per year / 365 = 2,739 images per day
    # 2739 x 2 = 5,478 (minutes to upscale, which assumes 2min to upscale each image)
    # 5478/60 = 91 (hours)
    # 91/24 = 3.8 (instances per hour)
    # 91/4 = 22.75 (22 hours to complete 1 days worth of images with 4 instances doing the work)

    daily_images = images_per_year / 365
    daily_minutes = daily_images * minutes_per_image
    daily_hours = daily_minutes / 60
    instances_needed = daily_hours / 24
    daily_hours_all_instances = daily_hours / instances_needed

    # round up instances_needed to nearest whole number
    instances_needed = int(instances_needed) + (instances_needed % 1 > 0)
    # round up daily_hours to nearest whole number
    daily_hours_all_instances = int(daily_hours_all_instances) + (daily_hours_all_instances % 1 > 0)

    return (instances_needed, daily_hours_all_instances)

# Usage example
instances_needed, daily_hours = quick_instance_calc(1000000, 2)
print(f"With {instances_needed} instances will take {daily_hours} hours to process 1 days worth of images")