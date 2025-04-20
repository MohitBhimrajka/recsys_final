// frontend/src/components/UserSelector.tsx
import React, { useState, useEffect } from 'react';
import Select, { SingleValue } from 'react-select'; // Import Select and SingleValue type
import { fetchUsers } from '../services/recommendationService';
import { User } from '../types';

// Define the structure for react-select options
interface UserOption {
  value: number;
  label: string;
}

interface UserSelectorProps {
  onUserSelect: (userId: number) => void;
  isLoading: boolean; // To disable selector during parent loading
}

const UserSelector: React.FC<UserSelectorProps> = ({ onUserSelect, isLoading }) => {
  const [userOptions, setUserOptions] = useState<UserOption[]>([]);
  const [selectedOption, setSelectedOption] = useState<SingleValue<UserOption>>(null);
  const [isFetchingUsers, setIsFetchingUsers] = useState<boolean>(true); // Separate loading state for users
  const [fetchError, setFetchError] = useState<string | null>(null);

  // Fetch users when the component mounts
  useEffect(() => {
    const loadUsers = async () => {
      setIsFetchingUsers(true);
      setFetchError(null);
      try {
        const users: User[] = await fetchUsers();
        if (users && users.length > 0) {
          const options = users.map(user => ({
            value: user.student_id,
            label: String(user.student_id) // Label must be a string
          }));
          setUserOptions(options);
        } else {
            setFetchError("No users found or failed to load user list.");
            setUserOptions([]); // Ensure options are empty on error
        }
      } catch (error) {
        console.error("Failed to load users:", error);
        setFetchError("Failed to load user list from API.");
        setUserOptions([]);
      } finally {
        setIsFetchingUsers(false);
      }
    };

    loadUsers();
  }, []); // Empty dependency array ensures this runs only once on mount

  // Handle selection change
  const handleChange = (selected: SingleValue<UserOption>) => {
    setSelectedOption(selected);
    if (selected) {
      onUserSelect(selected.value); // Pass the numeric user ID up
    }
  };

  return (
    <div className="mb-6 p-4 bg-white shadow rounded">
      <label htmlFor="userSelect" className="block text-sm font-medium text-gray-700 mb-2">
        Select Student ID:
      </label>
      <Select<UserOption> // Specify the type for Select
        id="userSelect"
        options={userOptions}
        value={selectedOption}
        onChange={handleChange}
        isLoading={isFetchingUsers} // Show loading indicator while fetching users
        isDisabled={isLoading || isFetchingUsers} // Disable if parent is loading OR if fetching users
        placeholder={isFetchingUsers ? "Loading users..." : "Type or select a student ID"}
        isClearable={true}
        isSearchable={true}
        noOptionsMessage={() => fetchError ? fetchError : "No users found"}
        styles={{ // Optional: basic styling adjustments
          control: (provided, state) => ({
            ...provided,
            borderColor: state.isFocused ? 'rgb(79 70 229)' : 'rgb(209 213 219)', // indigo-600, gray-300
            boxShadow: state.isFocused ? '0 0 0 1px rgb(79 70 229)' : 'none',
            '&:hover': {
                borderColor: state.isFocused ? 'rgb(79 70 229)' : 'rgb(156 163 175)', // gray-400
            }
          }),
          menu: (provided) => ({
              ...provided,
              zIndex: 20 // Ensure dropdown appears above other elements if needed
          })
        }}
      />
      {fetchError && !isFetchingUsers && ( // Show fetch error only after loading attempt
          <p className="mt-2 text-xs text-red-600">{fetchError}</p>
      )}
    </div>
  );
};

export default UserSelector;